# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in plaintext and performs
on-the-fly tokenization.
"""

from collections import defaultdict
import functools
import logging
import multiprocessing
import os
from metaseq import utils
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from metaseq.data import (
    Dictionary,
    JsonlDataset,
    PartitionedStreamingDataset,
    ResamplingDataset,
    StreamingDiffusionTokenBlockDatasetWithReplayBuffer,
    StreamingShuffleDataset,
    StreamingSrcTgtDataset,
    StreamingTokenBlockDataset,
    data_utils,
    iterators,
)
from metaseq.dataclass import MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task
from metaseq.tasks.streaming_language_modeling import (
    StreamingLanguageModelingConfig,
    StreamingLanguageModelingTask,
)

from metaseq.models.transformer import PositionalEmbedding

from omegaconf import II

try:
    from tokenizers import ByteLevelBPETokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


logger = logging.getLogger(__name__)

DEFAULT_MULTICORPUS_MAX = -1


@dataclass
class StreamingDiffusionLanguageModelingConfig(StreamingLanguageModelingConfig):
    max_T: int = field(
        default=4,
        metadata={"help": "max number of diffusion steps"},
    )
    select_T_strategy: Optional[str] = field(
        default="uniform", metadata={"help": "T distrbution"}
    )
    eviction_policy: str = field(
        default="random",
        metadata={
            "help": "policy by which to remove extra diffused steps from the buffer"
        },
    )
    max_buffer_size: int = field(
        default=200,
        metadata={"help": "maximum number of samples in the buffer"},
    )
    use_probabilistic_embedding_proj_rank: Optional[int] = field(
        default=-1, metadata={"help": "Top probabilities to take before projecting"}
    )
    full_context_alignment: Optional[bool] = field(
        default=False, metadata={"help": "Use full context aligment or not"}
    )
    step_positioning_policy: Optional[str] = field(
        default="",
        metadata={"help": "Diffusion step positioning policy: token, embedding, none"},
    )
    step_positioning_embedding_learned: Optional[bool] = field(
        default=False, metadata={"help": "Diffusion step PositionalEmbedding learned"}
    )
    step_positioning_embedding_learned_sinusoidal: Optional[bool] = field(
        default=False,
        metadata={"help": "Diffusion step PositionalEmbedding learned_sinusoidal"},
    )


@register_task(
    "streaming_diffusion_language_modeling",
    dataclass=StreamingDiffusionLanguageModelingConfig,
)
class StreamingDiffusionLanguageModelingTask(StreamingLanguageModelingTask):
    """
    Train a language model on a stream of data. Currently we assume the stream
    is in JSONL format and we tokenize inputs on-the-fly.

    Note that we append an end-of-document symbol to the end of each document.

    Args:
        tokenizer (tokenizers.ByteLevelBPETokenizer): the BPE tokenizer to use
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        if args.step_positioning_policy == "token":
            self._initialize_metaseq_dictionary(args)
        elif args.step_positioning_policy == "embedding":
            self.cfg_model = kwargs["cfg_model"]
            self._initialize_step_embeddings(args)

    def _initialize_metaseq_dictionary(self, args):
        self.step_tokens = []
        for t in range(args.max_T):
            self.step_tokens.append(self.dictionary.add_symbol(f"diff:{t}"))

        final_vocab_size = args.final_vocab_size
        # final_vocab_size = 51200 for roberta dictionary
        if final_vocab_size is not None:
            if final_vocab_size < tok_vocab_size:
                raise ValueError(
                    f"incompatible: {final_vocab_size}, tok_vocab_size: {tok_vocab_size}"
                )
            self.dictionary.pad_to_multiple_(final_vocab_size)
        else:
            self.dictionary.pad_to_multiple_(8)

    def _initialize_step_embeddings(self, args):
        self.step_embeddings = []
        for t in range(args.max_T):
            positional_embedding = PositionalEmbedding(
                len(self.dictionary),
                self.cfg_model.decoder_input_dim,
                self.dictionary.pad(),
                learned=args.step_positioning_embedding_learned,
                learned_sinusoidal=args.step_positioning_embedding_learned_sinusoidal,
            )

            initialize_params_on_gpu = getattr(
                self.cfg_model, "tensor_parallel_init_model_on_gpu", False
            )
            if initialize_params_on_gpu:
                positional_embedding = utils.floating_point_precision_convertor(
                    positional_embedding.cuda(),
                    fp16=getattr(args, "fp16", False),
                    memory_efficient_fp16=getattr(args, "memory_efficient_fp16", False),
                    bf16=getattr(args, "bf16", False),
                )

            self.step_embeddings.append(positional_embedding)
