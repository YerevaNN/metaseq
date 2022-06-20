import torch
import torch.nn as nn

from typing import Tuple


class ProbabilisticEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx,
        initialize_params_on_gpu=False,
    ) -> None:
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        dtype = torch.half if initialize_params_on_gpu else torch.float
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.normal_(weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(weight[padding_idx], 0)
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=weight
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        if not isinstance(x, tuple):
            return super().forward(x)
        flattened_prob, flattened_ind = x
        flattened_prob = flattened_prob.to(self.weight.device)
        flattened_ind = flattened_ind.to(self.weight.device)

        flattened_ind_batched = flattened_ind.view(-1, flattened_ind.size(2))
        flattened_prob_batched = flattened_prob.view(-1, flattened_prob.size(2))
        mapped = super().forward(flattened_ind_batched)
        mapped = mapped * flattened_prob_batched.unsqueeze(-1).to(mapped.dtype)
        mapped = mapped.sum(1)
        return mapped.view(
            flattened_ind.size(0), flattened_ind.size(1), self.weight.size(1)
        )
