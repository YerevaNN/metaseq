# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import csv
from tqdm import tqdm

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


@register_criterion("cross_entropy")
class CrossEntropyCriterion(BaseCriterion):
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
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if "src_tokens" in sample["net_input"] and hasattr(self.task, "eod"):
            logging_output["ndocseps"] = (sample["target"] == self.task.eod).sum()
        if (
            len(net_output) >= 2
            and isinstance(net_output[1], dict)
            and "inner_states" in net_output[1]
        ):
            with torch.no_grad():
                # yank out the inner states we wish to instrument
                # see transformer_decoder.py TransformerDecoder.extract_features
                emb, *_, actv = net_output[1]["inner_states"]
                assert isinstance(
                    emb, dict
                ), "Expecting the first inner state to be a dict of embedding representations"
                emb["actv"] = actv  # throw on final for code brevity
                for key, value in emb.items():
                    if value is None:
                        # maybe future proofing relative positional embeddings
                        continue
                    value = emb[key]
                    logging_output[f"{key}_norm"] = value.norm(p=2, dim=-1).sum(
                        dtype=torch.float32
                    )

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample).view(-1)
        loss = nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        return loss, loss
    
    def compute_perplexity(self, model, sample):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample).view(-1)

        perplexity_values = []
        target_chunk_list = []
        lprobs_chunk_list = []
        target_chunks_list = []

        # Shape: (batch_size * sequence_len) 
        target_list = target.data.tolist() 
        lprobs_list = lprobs.data.tolist() 

        for i in range(len(target_list)):
            if target_list[i] == self.task.target_dictionary.bos_index:
                # Start of the sequence
                target_chunk_list = []
                lprobs_chunk_list = []
            elif target_list[i] == self.task.target_dictionary.eos_index:
                # End of the sequence
                if target_chunk_list:
                    # Loss mean for one sequence through all its tokens
                    lprobs_chunk = torch.Tensor(lprobs_chunk_list)
                    target_chunk = torch.LongTensor(target_chunk_list)

                    nll_loss_one_seq = F.nll_loss(
                        lprobs_chunk,
                        target_chunk,
                        reduction="mean"
                    )

                    # Perplexity of a sequence
                    pp_seq = utils.get_perplexity(nll_loss_one_seq, 4)

                    # Row probability
                    row_probs = torch.exp(lprobs_chunk)
                    row_probs_max = torch.max(row_probs, dim=-1, keepdim=False).values
                    row_probs_likelihood = torch.prod(row_probs_max).item()

                    # NLL
                    nll_loss_one_seq = nll_loss_one_seq.item()

                    # Sequence decoded
                    target_seq = "".join([self.task.tokenizer.decode([t]) for t in target_chunk_list])

                    if len(self.task.dictionary.indices) < 200:
                        # if selfies
                        target_seq = utils.get_smiles_from_selfies(target_seq)
                    else: 
                         target_seq = utils.get_canonical_form(target_seq)

                    # Smiles, PP, NLL, Probability
                    vals = [target_seq, f'{pp_seq}', f'{nll_loss_one_seq:.4f}', f'{row_probs_likelihood:.8f}']

                    perplexity_values.append(vals)

                    # For checking
                    target_chunks_list.append(target_chunk_list)

                target_chunk_list = []
            else:
                target_chunk_list.append(target_list[i])        
                lprobs_chunk_list.append(lprobs_list[i]) 

            # For checking
            # print(target_chunks_list)           
            

        return perplexity_values


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for type_ in ("actv", "pos", "tok", "emb"):
            key = f"{type_}_norm"
            if any(key in log for log in logging_outputs):
                actv_norm = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, actv_norm / ntokens, round=4)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=4
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=4
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg, 4)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg, 4)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
