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
        default="", metadata={"help": "Diffusion step positioning policy: token, embedding, none"}
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

    def __init__(self, args):
        super().__init__(args)
        self._initialize_metaseq_dictionary(args)

    def _initialize_metaseq_dictionary(self, args):
        if args.step_positioning_policy == "token":
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
