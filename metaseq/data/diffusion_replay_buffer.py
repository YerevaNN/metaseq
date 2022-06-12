# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import List, Optional
import numpy as np
import functools
import torch

from . import BaseWrapperDataset, StreamingTokenBlockDataset


class BufferStack(list):
    def __init__(self, eviction_policy: str = "random", max_buffer_size: int = 20):
        self.eviction_policy = eviction_policy
        self.max_buffer_size = max_buffer_size

    def append(self, item):
        if self.eviction_policy == "random":
            return self.append_random(item)
        # elif self.eviction_policy == "stack":
        #     return self.append_stack(item)
        else:
            raise NotImplementedError(f"unkown eviction_policy: {self.eviction_policy}")

    def append_random(self, item):
        if len(self) + 1 > self.max_buffer_size:
            repl_id = np.random.randint(0, len(self))
            self[repl_id] = item
        else:
            super(BufferStack, self).append(item)

    def append_stack(self, item):
        pass


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
        eviction_policy: str = "random",
        max_buffer_size: int = 20,
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
        buffer_stack = functools.partial(
            BufferStack,
            eviction_policy=eviction_policy,
            max_buffer_size=max_buffer_size,
        )
        self.replay_buffer = defaultdict(buffer_stack)
        self.embedding_module = embedding_module
        self.split = split

    def sample_diffusion_step(self):
        return np.random.choice(
            list(range(len(self.diffusion_step_probabilities))),
            p=self.diffusion_step_probabilities,
        )

    def sample_diffusion_step_existing(self):
        T = self.sample_diffusion_step()
        # TODO (?)
        while len(self.replay_buffer[T]) == 0 and T > 0:
            T = self.sample_diffusion_step()
        return T

    def item_input(self, item, device):
        x = item[:-1].to(device).unsqueeze(0).contiguous()
        return x

    def __iter__(self):
        for item in super().__iter__():
            if item is None:
                yield None
                continue

            T = self.sample_diffusion_step_existing()
            device = self.embedding_module.weight.device
            if T == 0:
                inp = self.item_input(item["block"], device)
                yield {
                    "T": 0,
                    "block": item["block"],
                    "ids": item["ids"],
                    "token_embeddings": self.embedding_module(inp).squeeze().cpu(),
                    "split": self.split,
                }
            else:
                item_idx = np.random.randint(0, len(self.replay_buffer[T]))
                item_diff = self.replay_buffer[T][item_idx]
                # TODO: check this logic
                if self.embedding_module.weight.dtype == torch.float16:
                    item_diff["probs"] = item_diff["probs"].half()
                yield {
                    "T": T,
                    "block": item_diff["block"],
                    "ids": item_diff["ids"],
                    "token_embeddings": (
                        item_diff["probs"].to(device) @ self.embedding_module.weight
                    ).cpu(),
                    "split": self.split,
                }
                del self.replay_buffer[T][item_idx]

    def update_buffer_batch(self, T: torch.Tensor, probs: torch.Tensor, item: dict):
        T_item = T[0].cpu().detach().item()
        for i in range(len(T)):
            self.replay_buffer[T_item].append(
                {
                    "block": item["block"][i].cpu().detach(),
                    "ids": item["id"][i].cpu().detach(),
                    "probs": probs[i].cpu().detach(),
                }
            )
