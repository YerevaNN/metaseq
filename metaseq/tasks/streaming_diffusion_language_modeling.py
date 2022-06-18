# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in plaintext and performs
on-the-fly tokenization.
"""

import logging
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
    StreamingDiffusionTokenBlockDatasetWithReplayBufferV2,
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
        default=20,
        metadata={"help": "maximum number of samples in the buffer"},
    )
    use_probabilistic_embedding_proj_rank: Optional[int] = field(
        default=-1, metadata={"help": "Top probabilities to take before projecting"}
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
        self.max_T = args.max_T

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        The folder structure is assumed to look like:

            /path/to/data/train/00/foo.jsonl
            /path/to/data/train/00/bar.jsonl
            /path/to/data/train/01/foo.jsonl
            /path/to/data/train/01/bar.jsonl
            /path/to/data/valid/00/foo.jsonl
            /path/to/data/valid/00/bar.jsonl

        In this example, we have two "shards" of training data, which will be
        iterated over in epochs 1 and 2, respectively. Subsequent epochs will
        cycle back over the same data. We also have two different data sources
        in each shard (foo and bar), which will be combined and shuffled.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).

        # determine number of shards for this split
        cur_shard_str = self.get_shard_str(epoch, split)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        for file in sorted(
            os.listdir(os.path.join(self.args.data, split, cur_shard_str))
        ):
            if not file.endswith(".jsonl"):
                continue
            datasets.append(
                JsonlDataset(
                    path=os.path.join(self.args.data, split, cur_shard_str, file),
                    tokenizer=self._tokenize_one_json,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if self.args.multicorpus_sampling_alpha != 1:
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        # shuffle order across epochs
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)

        # chunk into blocks of tokens
        pdf = None
        if self.args.select_T_strategy == "uniform":
            pdf = [1.0 / self.args.max_T] * self.args.max_T
        else:
            raise Exception
        self.datasets[split] = StreamingDiffusionTokenBlockDatasetWithReplayBufferV2(
            pdf,
            len(self.source_dictionary),
            split,
            dataset,
            use_probabilistic_embedding_proj_rank=self.args.use_probabilistic_embedding_proj_rank,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode=self.args.sample_break_mode,
            # we drop the remainder block during training
            drop_last=(split == "train"),
            padding_idx=self.source_dictionary.pad(),
            # 1284 is a randomly-generated offset to decouple the seed used here
            # from the seed used above in StreamingShuffleDataset
            seed=1284 + self.args.seed,
            eviction_policy=self.args.eviction_policy,
            max_buffer_size=self.args.max_buffer_size,
        )

    def build_model(self, args: Namespace):
        """
        Build the :class:`~metaseq.models.BaseModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~metaseq.models.BaseModel` instance
        """
        from metaseq import models

        self.model = models.build_model(args, self)
        return self.model

    def _collate_fn(self, items: List[Dict[str, Any]]):
        if len([x for x in items if x is not None]) == 0:
            return {}

        tokens = data_utils.collate_tokens(
            [x["block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )
        # generate inputs and targets
        input = tokens[:, :-1].contiguous()
        target = tokens[:, 1:].contiguous()

        ids = torch.cat([torch.tensor([0]) for x in items if x is not None])
        flattened_prob = torch.stack([x["probs"][0].cpu().detach() for x in items if x is not None])
        flattened_ind = torch.stack([x["probs"][1].cpu().detach() for x in items if x is not None])
        timesteps = torch.tensor([x["T"] for x in items if x is not None])
        split = [x["split"] for x in items if x is not None]
        
        # metaseq expects batches to have the following structure
        return {
            "id": ids,
            "net_input": {
                "src_tokens": input,
                "token_probs": (flattened_prob, flattened_ind),
                "full_context_alignment": torch.all(timesteps > 0).item(),
            },
            "target": target,
            "block": tokens,
            "T": timesteps,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.dictionary.pad()).sum(),
            "split": split,
        }
