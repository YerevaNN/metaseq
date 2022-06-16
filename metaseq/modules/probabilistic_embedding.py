import torch
import torch.nn as nn


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

        self.partial_projection_size = partial_projection

    def forward(self, x: torch.Tensor):
        if self.partial_projection_size < 0:
            return x @ self.weight
        return self.partial_projection(x, self.partial_projection_size)

    def partial_projection(self, probs, n: int):
        values = []
        for prob in probs:
            if (prob == 0).long().sum() > n:
                max_elem, max_elem_ind = torch.topk(prob, 1, dim=-1)
                values.append(super().forward(max_elem_ind).squeeze())
            else:
                max_elem, max_elem_ind = torch.topk(prob, n, dim=-1)

                partial_embed = self.weight[max_elem_ind]
                values.append((max_elem @ partial_embed)[:, 0])
        return torch.stack(values)
