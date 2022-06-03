# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Optional
import numpy as np
import torch

from . import BaseWrapperDataset, StreamingTokenBlockDataset


class DiffusionReplayBufferDataset(BaseWrapperDataset):
    def __init__(self, dataset, diffusion_step_probabilities: List[float], embedding_module: torch.nn.Module, split: str):
        super().__init__(dataset)
        self.diffusion_step_probabilities = diffusion_step_probabilities
        self.replay_buffer = defaultdict(list)
        # collections.deque(maxlen=5)
        self.embedding_module = embedding_module
        self.split = split

    def sample_diffusion_step(self):
        return np.random.choice(list(range(len(self.diffusion_step_probabilities))), 1, p=self.diffusion_step_probabilities)

    def __getitem__(self, idx):
        T = self.sample_diffusion_step()
        while len(self.replay_buffer[T]) == 0 and T > 0:
            T = self.sample_diffusion_step()

        if T == 0:
            item = self.dataset[idx]
            return {"T": 0, "block": item["block"], "ids": item["ids"], "token_embeddings": self.embedding_module(item["src_tokens"]), "split": self.split}
        # with some seed
        item = np.random.choice(self.replay_buffer[T])
        return {"T": T, "block": item["block"], "ids": item["ids"], "token_embeddings": item['probs'] @ self.embedding_module.embed_tokens.weight, "split": self.split}

    def update_buffer(self, T: int, probs: torch.Tensor, item: dict):
        self.replay_buffer[T].append({"block": item["block"], "ids": item["ids"], "probs": probs})


class StreamingDiffusionTokenBlockDatasetWithReplayBuffer(StreamingTokenBlockDataset):
    def __init__(self, diffusion_step_probabilities: List[float],
                 embedding_module: torch.nn.Module,
                 split: str,
                 dataset: torch.utils.data.IterableDataset,
                 block_size: int,
                 break_mode: str = "none",
                 drop_last: Optional[bool] = False,
                 padding_idx: Optional[int] = None,
                 shuffle_buffer_size: int = 1,
                 seed: Optional[int] = None):
        super().__init__(dataset, block_size, break_mode, drop_last, padding_idx, shuffle_buffer_size, seed)
        self.diffusion_step_probabilities = diffusion_step_probabilities
        self.replay_buffer = defaultdict(list)
        # collections.deque(maxlen=5)
        self.embedding_module = embedding_module
        self.split = split

    def sample_diffusion_step(self):
        return np.random.choice(list(range(len(self.diffusion_step_probabilities))), 1, p=self.diffusion_step_probabilities)

    def __getitem__(self, idx):
        print("HITTING")
        T = self.sample_diffusion_step()
        while len(self.replay_buffer[T]) == 0 and T > 0:
            T = self.sample_diffusion_step()

        if T == 0:
            item = super()[idx]
            if item is None:
                return None
            return {"T": 0, "block": item["block"], "ids": item["ids"], "token_embeddings": self.embedding_module(item["src_tokens"]), "split": self.split}
        # with some seed
        item = np.random.choice(self.replay_buffer[T])
        return {"T": T, "block": item["block"], "ids": item["ids"], "token_embeddings": item['probs'] @ self.embedding_module.embed_tokens.weight, "split": self.split}

    def update_buffer(self, T: int, probs: torch.Tensor, item: dict):
        self.replay_buffer[T].append({"block": item["block"], "ids": item["ids"], "probs": probs})
