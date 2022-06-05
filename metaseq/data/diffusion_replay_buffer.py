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
    def __init__(
        self,
        dataset,
        diffusion_step_probabilities: List[float],
        embedding_module: torch.nn.Module,
        split: str,
    ):
        super().__init__(dataset)
        self.diffusion_step_probabilities = diffusion_step_probabilities
        self.replay_buffer = defaultdict(list)
        # collections.deque(maxlen=5)
        self.embedding_module = embedding_module
        self.split = split

    def sample_diffusion_step(self):
        return np.random.choice(
            list(range(len(self.diffusion_step_probabilities))),
            1,
            p=self.diffusion_step_probabilities,
        )

    def __getitem__(self, idx):
        T = self.sample_diffusion_step()
        while len(self.replay_buffer[T]) == 0 and T > 0:
            T = self.sample_diffusion_step()

        if T == 0:
            item = self.dataset[idx]
            return {
                "T": 0,
                "block": item["block"],
                "ids": item["ids"],
                "token_embeddings": self.embedding_module(item["src_tokens"]),
                "split": self.split,
            }
        # with some seed
        item = np.random.choice(self.replay_buffer[T])
        return {
            "T": T,
            "block": item["block"],
            "ids": item["ids"],
            "token_embeddings": item["probs"]
            @ self.embedding_module.embed_tokens.weight,
            "split": self.split,
        }

    def update_buffer(self, T: int, probs: torch.Tensor, item: dict):
        self.replay_buffer[T].append(
            {"block": item["block"], "ids": item["ids"], "probs": probs}
        )


class StreamingDiffusionTokenBlockDatasetWithReplayBuffer(StreamingTokenBlockDataset):
    def __init__(
        self,
        diffusion_step_probabilities: List[float],
        embedding_module: torch.nn.Module,
        split: str,
        dataset: torch.utils.data.IterableDataset,
        block_size: int,
        break_mode: str = "none",
        drop_last: Optional[bool] = False,
        padding_idx: Optional[int] = None,
        shuffle_buffer_size: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(
            dataset,
            block_size,
            break_mode,
            drop_last,
            padding_idx,
            shuffle_buffer_size,
            seed,
        )
        self.diffusion_step_probabilities = diffusion_step_probabilities
        self.replay_buffer = defaultdict(list)
        # collections.deque(maxlen=5)
        self.embedding_module = embedding_module
        self.split = split

    def sample_diffusion_step(self):
        return np.random.choice(
            list(range(len(self.diffusion_step_probabilities))),
            p=self.diffusion_step_probabilities,
        )

    def sample_diffusion_step_existing(self):
        T = self.sample_diffusion_step()
        while len(self.replay_buffer[T]) == 0 and T > 0:
            T = self.sample_diffusion_step()
        return T

    def item_input(self, item, device):
        x = item[:-1].to(device).unsqueeze(0).contiguous()
        print(x.size())
        return x

    def __iter__(self):
        for item in super().__iter__():
            with torch.no_grad():
                T = self.sample_diffusion_step_existing()
                device = self.embedding_module.weight.device
                print(device)
                print(self.embedding_module(torch.randn((1, 2048)).long().to(device)))
                if T == 0:
                    if item is None:
                        yield None
                    else:
                        yield {
                            "T": 0,
                            "block": item["block"],
                            "ids": item["ids"],
                            "token_embeddings": self.embedding_module(
                                self.item_input(item["block"], device)
                            ).squeeze(),
                            "split": self.split,
                        }
                else:
                    while T != 0:
                        # with some seed
                        item_diff = np.random.choice(self.replay_buffer[T])
                        yield {
                            "T": T,
                            "block": item_diff["block"],
                            "ids": item_diff["ids"],
                            "token_embeddings": item_diff["probs"].to(device)
                            @ self.embedding_module.weight,
                            "split": self.split,
                        }
                        T = self.sample_diffusion_step_existing()
                    if item is None:
                        yield None
                    else:
                        yield {
                            "T": 0,
                            "block": item["block"],
                            "ids": item["ids"],
                            "token_embeddings": self.embedding_module(
                                self.item_input(item["block"], device)
                            ).squeeze(),
                            "split": self.split,
                        }

    def update_buffer_batch(self, T: torch.Tensor, probs: torch.Tensor, item: dict):
        T_item = T[0].cpu().detach().item()
        print(item)
        for i in range(len(T)):
            self.replay_buffer[T_item].append(
                {
                    "block": item["block"][i].cpu().detach(),
                    "ids": item["ids"][i].cpu().detach(),
                    "probs": probs[i].cpu().detach(),
                }
            )
