# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# metaseq implementation of
# https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import math

import torch
import torch.nn.functional as F

from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion


def nll_loss(lprobs, target, ignore_index=None, reduction="mean"):
    """Like torch.nn.functional.nll_loss but works for large inputs."""
    if lprobs.numel() < 2e9:
        return F.nll_loss(
            lprobs, target, ignore_index=ignore_index, reduction=reduction
        )
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduction == "mean":
        nll_loss = nll_loss.mean()
    elif reduction == "sum":
        nll_loss = nll_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError
    return nll_loss


@register_criterion("ddpm_cross_entropy")
class DDPMCrossEntropyCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        diffused_losses = {f"diff_loss_{t}": 0.0 for t in range(self.task.args.max_T)}
        sample_size = sample["ntokens"]
        logging_output = {
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        logging_output.update(diffused_losses)

        prev_input = sample["net_input"]
        prev_input["is_ddpm"] = True
        total_loss = 0
        for T in range(self.task.args.max_T):
            if self.task.args.step_positioning_policy == "token":
                positional_tokens = torch.tensor(
                    [
                        self.task.step_tokens[T]
                        for _ in range(prev_input["src_tokens"].shape[0])
                    ],
                    device=prev_input["src_tokens"].device,
                ).unsqueeze(1)
                prev_input["src_tokens"] = torch.cat(
                    (positional_tokens, prev_input["src_tokens"][:, :-1]), dim=1
                )
            elif self.task.args.step_positioning_policy == "embedding":
                prev_input["diff_embed_positions"] = self.task.step_embeddings[T]
            net_output = model(**prev_input)
            loss, unreduced_loss, probs = self.compute_loss(
                model, net_output, sample, reduce=reduce
            )
            total_loss = total_loss + loss
            logging_output[f"diff_loss_{T}"] = loss.data
            logging_output[f"diff_loss_{T}_size"] = unreduced_loss.numel()
            current_rank = None
            if self.task.args.use_probabilistic_embedding_proj_rank_min == -1:
                current_rank = probs.shape[-1]
            else:
                current_rank = int(
                    self.task.args.use_probabilistic_embedding_proj_rank_max
                    - T
                    * (
                        (
                            self.task.args.use_probabilistic_embedding_proj_rank_max
                            - self.task.args.use_probabilistic_embedding_proj_rank_min
                        )
                        / self.task.args.max_T
                    )
                )
                assert current_rank > 0

            flattened_prob, flattened_ind = torch.topk(
                probs.detach(),
                current_rank,
                dim=-1,
            )
            prev_input = {
                "src_tokens": prev_input["src_tokens"],
                "token_probs": (flattened_prob, flattened_ind),
                "full_context_alignment": self.task.args.full_context_alignment,
                "is_ddpm": True
            }
        loss = total_loss / self.task.args.max_T
        logging_output["loss"] = loss.data

        if "src_tokens" in sample["net_input"] and hasattr(self.task, "eod"):
            logging_output["ndocseps"] = (sample["target"] == self.task.eod).sum()
        if (
            len(net_output) >= 2
            and isinstance(net_output[1], dict)
            and "inner_states" in net_output[1]
        ):
            with torch.no_grad():
                # yank out the inner states we wish to instrument
                # see transformer.py TransformerDecoder.extract_features_scriptable
                emb, *_, actv = net_output[1]["inner_states"]
                assert isinstance(
                    emb, dict
                ), "Expecting the first inner state to be a dict of embedding representations"
                emb["actv"] = actv  # throw on final for code brevity
                for key, value in emb.items():
                    if value is None:
                        # maybe future proofing relative positional embeddings
                        continue
                    logging_output[f"{key}_norm"] = value.norm(p=2, dim=-1).sum(
                        dtype=torch.float32
                    )
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)

        lprobs = torch.log(probs)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="none",
        )
        return loss.sum() if reduce else loss, loss, probs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        diffused_losses_sum = {}
        for i in range(5):
            key = f"diff_loss_{i}"
            if any(key in log for log in logging_outputs):
                diffused_losses_sum[key] = (
                    sum(log.get(key, 0) for log in logging_outputs),
                    sum(log.get(f"{key}_size", 0) for log in logging_outputs),
                )

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for type_ in ("actv", "pos", "tok", "emb"):
            key = f"{type_}_norm"
            if any(key in log for log in logging_outputs):
                actv_norm = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, actv_norm / ntokens, round=3)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        for key, (value, key_sample_size) in diffused_losses_sum.items():
            if key_sample_size != 0:
                metrics.log_scalar(
                    key, value / key_sample_size / math.log(2), key_sample_size, round=3
                )
                metrics.log_scalar(
                    f"diff_ppl_{key[-1]}",
                    torch.pow(2, value / key_sample_size / math.log(2)),
                    round=3,
                )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
