# Run the following to get a single consolidated checkpoint file e.g. `checkpoint_last_consolidated.pt`
"""
model_dir=<checkpoint_parent_directory>
python -m metaseq.scripts.consolidate_fsdp_shards $model_dir/checkpoint_last --new_arch_name transformer_lm
"""

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional, Callable
from metaseq import (
    options,
    utils,
)
from metaseq.models.transformer_lm import TransformerLanguageModel
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
import torch
from torch import nn
from typing import Dict, Optional
from omegaconf import DictConfig
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(cfg: DictConfig) -> None:
    utils.import_user_module(cfg.common)

    model = TransformerLanguageModel.from_pretrained(
        cfg.checkpoint.save_dir,
        cfg.checkpoint.restore_file,
    )

    model.binarize = binarize
    # model = model.cuda().eval().float()

    tokenizer = ByteLevelBPETokenizer.from_file(cfg.criterion.vocab_filename, cfg.criterion.merges_filename)

    T0_sampler = TopPSampling(model, 256, 1024 + 256, 0.25, 1, True)
    TN_sampler = GreedyDiffusionSampling(model, 15)
    T0_sampler.cuda().half()
    TN_sampler.cuda().half()

    prompt = binarize(tokenizer, "they")
    decoded_tokens, log_probs = T0_sampler.decode(prompt)
    output = TN_sampler.diffuse(model, log_probs, decoded_tokens)

    print(decode_tokenizer(tokenizer, output[0]))


def binarize(tokenizer, sentence: str) -> torch.LongTensor:
    return torch.LongTensor(tokenizer.encode(sentence).ids).half()


def forward_encoder(model, net_input):
    if not hasattr(model, "encoder"):
        return None
    return model.encoder.forward_torchscript(net_input)


def decode_tokenizer(tokenizer, tokenized_sent: torch.LongTensor):
    return " ".join([tokenizer.id_to_token(t) for t in tokenized_sent if tokenizer.id_to_token(t)]).replace("Ġ", " ")


def unpack_decoder_out(model, decoder_out, temperature: float):
    attn: Optional[torch.Tensor] = None
    decoder_len = len(decoder_out)
    if decoder_len > 1 and decoder_out[1] is not None:
        if isinstance(decoder_out[1], torch.Tensor):
            attn = decoder_out[1]
        else:
            attn_holder = decoder_out[1]["attn"]
            if isinstance(attn_holder, torch.Tensor):
                attn = attn_holder
            elif attn_holder is not None:
                attn = attn_holder[0]
        if attn is not None:
            attn = attn[:, -1, :]

    decoder_out, extras = (
        decoder_out[0][:, -1:, :].div_(temperature),
        None if decoder_len <= 1 else decoder_out[1],
    )
    probs = model.get_normalized_probs(decoder_out, log_probs=True)
    # probs = probs[:, -1, :]
    return probs, attn


class DecodingBase(nn.Module):
    def __init__(
        self,
        transformer_language_model,
        min_len: int,
        max_len: int,
        temperature: float = 1.0,
        return_probs: bool = False,
    ):
        super().__init__()
        self.language_model = transformer_language_model
        self.dictionary = transformer_language_model.task.source_dictionary
        self.model = transformer_language_model.models[0]
        self.model.cuda()
        self.eos = self.dictionary.eos()
        self.pad = self.dictionary.pad()
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.return_probs = return_probs

    def decode(self, prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.inference_mode():
            if prefix is None:
                prefix = torch.tensor([self.eos]).to(self.dummy_param.device)
            prefix = prefix.to(self.dummy_param.device)
            assert prefix.ndim == 1 and prefix.size(0) > 0

            if prefix[0] != self.eos:
                prefix = torch.cat([torch.tensor([self.eos]).to(prefix), prefix])
            prefix_len: int = prefix.size(0)
            assert prefix_len < self.max_len, "Max len is smaller than prefix length"

            src_tokens = prefix
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum()
            )
            encoder_out = forward_encoder(
                self.model,
                {
                    "src_tokens": src_tokens.unsqueeze(0),
                    "src_lengths": src_lengths.unsqueeze(0),
                },
            )

            incremental_states = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            tokens = (
                torch.zeros(1, self.max_len)
                .to(src_tokens.device)
                .long()
                .fill_(self.pad)
            )
            tokens[:, : len(prefix)] = prefix
            all_probs = []
            for step in tqdm(range(1, self.max_len + 1)):
                decoder_out = self.model.decoder.forward(
                    tokens[:, :step],
                    encoder_out=encoder_out,
                    incremental_state=incremental_states,
                )
                logprobs, _ = unpack_decoder_out(self.model, decoder_out, self.temperature)
                all_probs.append(logprobs)
                if step + 1 < self.max_len:
                    if step < len(prefix):
                        tokens[:, step] = prefix[step]
                    else:
                        tokens[:, step] = self.choice(logprobs.squeeze(), step)

            if self.return_probs:
                return tokens, torch.stack(all_probs, dim=1)
            return tokens, None

    def choice(self, logprob: torch.Tensor, step: int) -> int:
        raise NotImplementedError


class GreedyDecoding(DecodingBase):
    def choice(self, logprob: torch.Tensor, step: int) -> int:
        return torch.argmax(logprob)


class TopPSampling(DecodingBase):
    def __init__(
        self,
        transformer_language_model,
        min_len: int,
        max_len: int,
        sampling_topp: float,
        temperature: float = 1.0,
        return_probs: bool = False,
    ):
        super().__init__(
            transformer_language_model, min_len, max_len, temperature, return_probs
        )
        self.sampling_topp = sampling_topp

    def choice(self, logprob: torch.Tensor, step: int) -> int:
        probs = logprob.exp_()
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=0)
        mask = cumsum_probs.lt(self.sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=0)
        last_included = cumsum_mask[-1:]
        last_included.clamp_(0, mask.size(0) - 1)
        mask = mask.scatter_(0, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[: max_dim + 1]
        truncated_probs = sorted_probs[: max_dim + 1]
        truncated_indices = sorted_indices[: max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)

        indices_buf = torch.multinomial(trimed_probs, num_samples=1)
        return truncated_indices[indices_buf]


class TopKSampling(DecodingBase):
    def __init__(
        self,
        transformer_language_model,
        min_len: int,
        max_len: int,
        sampling_topk: int,
        temperature: float = 1.0,
        return_probs: bool = False,
    ):
        super().__init__(
            transformer_language_model, min_len, max_len, temperature, return_probs
        )
        self.sampling_topk = sampling_topk

    def choice(self, logprob: torch.Tensor, step: int) -> int:
        lprobs, top_indices = logprob.topk(self.sampling_topk)
        probs = lprobs.exp_()
        indices_buf = torch.multinomial(
            probs,
            1,
            replacement=True,
        )

        return top_indices[indices_buf]


class DiffusionSampling(nn.Module):
    def __init__(
        self,
        transformer_language_model,
        projection_rank: int,
        max_T: int = 5,
    ):
        super().__init__()

        assert projection_rank > 0

        self.projection_rank = projection_rank
        self.language_model = transformer_language_model
        self.dictionary = transformer_language_model.task.source_dictionary
        self.model = transformer_language_model.models[0]
        self.model.cuda()
        self.eos = self.dictionary.eos()
        self.pad = self.dictionary.pad()
        self.max_T = max_T
        self.dummy_param = nn.Parameter(torch.empty(0))

    def diffuse(self, model, prev_log_prob: torch.Tensor, prev_tokens: torch.Tensor):
        with torch.inference_mode():
            for _ in range(self.max_T):
                flattened_prob, flattened_ind = torch.topk(
                    prev_log_prob.detach(),
                    self.projection_rank,
                    dim=-1,
                )
                self_attn_padding_mask = torch.eq(prev_tokens, torch.zeros_like(
                    prev_tokens).fill_(model.task.dictionary.pad()))

                prev_log_prob, prev_extras = self.model.decoder.forward(prev_log_prob,
                                                                        token_probs=(flattened_prob, flattened_ind),
                                                                        full_context_alignment=True,
                                                                        self_attn_padding_mask=self_attn_padding_mask
                                                                        )
        return self.choice(prev_log_prob).cpu().numpy().tolist()

    def choice(self, logprob: torch.Tensor, step: int) -> int:
        raise NotImplementedError


class GreedyDiffusionSampling(DiffusionSampling):
    def choice(self, logprob: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logprob, dim=-1)


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    main(cfg)


if __name__ == "__main__":
    cli_main()
