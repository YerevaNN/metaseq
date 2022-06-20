# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


@register_criterion("diffusion_cross_entropy")
class DiffusionCrossEntropyCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, unreduced_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce
        )
        sample_size = sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        diffused_losses = {f"diff_loss_{t}": 0.0 for t in range(self.task.max_T)}
        logging_output.update(diffused_losses)

        unreduced_loss_reshaped = torch.reshape(unreduced_loss, (len(sample["T"]), -1))
        for t in range(max(sample["T"]) + 1):
            indices = torch.tensor(
                [i for i in range(len(sample["T"])) if bool(sample["T"][i] == t)]
            )
            if indices.numel() == 0:
                continue
            t_loss = torch.index_select(unreduced_loss_reshaped.cpu(), 0, indices)
            logging_output[f"diff_loss_{t}"] = t_loss.flatten().detach().sum()
            logging_output[f"diff_loss_{t}_size"] = t_loss.numel()

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
        # Add to replay buffer
        # Figure out a way to pass split
        self.task.update_buffer_batch(
            sample["T"], probs.cpu(), sample, sample["split"][0]
        )

        lprobs = torch.log(probs)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="none",
        )
        return loss.sum() if reduce else loss, loss

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


@register_criterion("diffusion_cross_entropy_balanced")
class DiffusionCrossEntropyBalancedCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)

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
        total_loss = 0
        for T in range(self.task.args.max_T):
            net_output = model(**prev_input)
            loss, unreduced_loss, probs = self.compute_loss(
                model, net_output, sample, reduce=reduce
            )
            total_loss = total_loss + loss
            logging_output[f"diff_loss_{T}"] = loss.data
            logging_output[f"diff_loss_{T}_size"] = unreduced_loss.numel()
            flattened_prob, flattened_ind = torch.topk(
                probs.detach(),
                self.task.args.use_probabilistic_embedding_proj_rank,
                dim=-1,
            )
            prev_input = {
                "src_tokens": prev_input["src_tokens"],
                "token_probs": (flattened_prob, flattened_ind),
                "full_context_alignment": False,
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
