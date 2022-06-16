import torch
import torch.nn as nn


def partial_projection(probs, embedding_weight, n: int):
    max_elem, max_elem_ind = torch.topk(probs, n)
    partial_embed = embedding_weight[max_elem_ind]
    return max_elem @ partial_embed


class ProbabilisticEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx,
        partial_projection: int = -1,
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

        self.partial_projection = partial_projection

    def forward(self, x: torch.Tensor):
        if self.partial_projection < 0:
            return x @ self.weight
        return partial_projection(x, self.weight, self.partial_projection)
