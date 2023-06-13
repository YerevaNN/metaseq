# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import logging
import math
from typing import Dict, Iterator, List, Optional
import tqdm
import torch
import torch.nn as nn
from torch import Tensor
import gradio as gr

logger = logging.getLogger(__name__)

IMAGE_TOKENS_COUNT = 1024

GeneratorIteratorState = namedtuple(
    "GeneratorIteratorState", ["logits", "scores", "tokens"]
)


def map_old_image_token_to_new_image_token(text):
    text = text.replace("I", "IMGIMG")
    for i in range(10):
        text = text.replace(str(i), chr(ord("A") + i))
    return text.replace(" ", "Z")


def map_new_image_token_to_old_image_token(text):
    text = text.replace("Z", "")
    for i in range(10):
        text = text.replace(chr(ord("A") + i), str(i))
    return text.replace("IMGIMG", "I")


def extract_image_tokens(text):
    tokens = []
    for x in text.split("IMGIMG"):
        try:
            tokens.append(int(map_new_image_token_to_old_image_token(x)))
        except:
            continue
    tokens = tokens[:1024]
    if len(tokens) < 1024:
        tokens = tokens + [0] * (1024 - len(tokens))
    return tokens


@torch.jit.script
def _sample_topp(temperature: float, sampling_topp: float, lprobs: torch.Tensor):
    if temperature == 0.0 or sampling_topp == 0.0:
        # greedy search
        return tuple(lprobs.max(dim=-1))

    probs = lprobs.exp()
    sprobs, sinds = probs.sort(dim=-1, descending=True)
    mask = (sprobs.cumsum(dim=-1) - sprobs) >= sampling_topp
    trunc_sprobs = sprobs.detach().clone()
    trunc_sprobs[mask] = 0
    trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(-1))
    choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
    hyp_ids = torch.arange(lprobs.size(0)).to(lprobs.device)
    tok_ids = sinds[hyp_ids, choices]
    scores = sprobs[hyp_ids, choices].log()
    return scores, tok_ids


@torch.jit.script
def _classifier_free_guidance(
    cfg_weight: float, unconditioned_logits: Tensor, conditioned_logits: Tensor
):
    return unconditioned_logits + cfg_weight * (
        conditioned_logits - unconditioned_logits
    )


class ImageSequenceGenerator(nn.Module):
    def __init__(
        self,
        model,
        tgt_dict,
        progress: gr.Progress,
        beam_size: int = 1,
        temperature: float = 1.0,
        stop: Optional[List[int]] = None,
        topp: float = 1,
        cfg_weight: float = 3.0,
        replicate: int = 32,
    ):
        super().__init__()
        self.model = model
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.stop = stop if stop is not None else []
        if topp is None:
            topp = 0.0
        self.sampling_topp = max(0, topp)
        self.temperature = temperature
        assert temperature >= 0, "--temperature must be >=0"

        self.model.eval()
        self.non_image_tokens = torch.tensor(
            [ind for x, ind in self.tgt_dict.indices.items() if "IMGIMG" not in x]
        )
        self.image_tokens = torch.tensor(
            [ind for x, ind in self.tgt_dict.indices.items() if "IMGIMG" in x]
        )
        self.progress = progress
        self.cfg_weight = cfg_weight
        self.replicate = replicate
        self.dummy_param = nn.Parameter(torch.empty(0))

    @torch.inference_mode()
    def mask_non_image(self, logits):
        logits[:, self.non_image_tokens.to(logits.device)] = -100
        return logits

    @torch.inference_mode()
    def mask_non_text(self, logits):
        logits[:, self.image_tokens.to(logits.device)] = -100
        return logits

    @torch.inference_mode()
    def mask_special_tokens(self, step, lprobs, min_len):
        if step < min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            lprobs[:, self.eos] = -math.inf
            for stop_token in self.stop:
                lprobs[:, stop_token] = -math.inf

        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
        lprobs[:, self.pad] = -math.inf  # never select pad
        return lprobs

    @torch.inference_mode()
    def forward(self, src_tokens: Tensor, src_tokens_unconditional: Tensor):
        src_tokens = src_tokens.to(self.dummy_param.device)
        src_tokens_unconditional = src_tokens_unconditional.to(self.dummy_param.device)

        bsz, _ = src_tokens.size()[:2]
        scores = (
            torch.zeros(bsz * self.beam_size, IMAGE_TOKENS_COUNT + 1)
            .to(src_tokens)
            .float()
        )
        tokens = (
            torch.zeros(bsz * self.beam_size, IMAGE_TOKENS_COUNT + 1)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )
        self.shared_state: List[GeneratorIteratorState] = []
        src_token_iterator = self.generate_iterator(src_tokens)
        src_tokens_unconditional_iterator = self.generate_iterator(
            src_tokens_unconditional
        )
        for step in self.progress.tqdm(range(IMAGE_TOKENS_COUNT)):
            # Can't use zip because of gradio progress bar
            conditional_logit = next(src_token_iterator)
            unconditional_logit = next(src_tokens_unconditional_iterator)
            # Keep Only Image Tokens
            conditional_logit = self.mask_non_image(conditional_logit)
            unconditional_logit = self.mask_non_image(unconditional_logit)
            # Do Classifier Free Guidance
            cfg_mixed_logits = _classifier_free_guidance(
                self.cfg_weight,
                conditioned_logits=conditional_logit,
                unconditioned_logits=unconditional_logit,
            )
            # Apply temperature and get log probabilities
            cfg_mixed_logits.div_(self.temperature)
            cfg_mixed_lprobs = self.model.get_normalized_probs(
                cfg_mixed_logits, log_probs=True
            )
            # Mask out EOS, PAD tokens
            cfg_mixed_lprobs = self.mask_special_tokens(
                step, cfg_mixed_lprobs, IMAGE_TOKENS_COUNT
            )
            # Sample the next token
            next_scores, next_toks = _sample_topp(
                self.temperature, self.sampling_topp, cfg_mixed_lprobs
            )
            tokens[:, step] = next_toks
            scores[:, step] = next_scores
            # Update the shared state so that sub-iterators can replace their tokens
            self.shared_state.append(
                GeneratorIteratorState(cfg_mixed_logits, next_scores, next_toks)
            )
        # we want the highest scoring items to be top ranked
        beamscores = scores.view(bsz, self.beam_size, -1).cumsum(dim=-1)[:, :, -1]
        indices = beamscores.sort(dim=-1, descending=True).indices
        sorted_indices = (
            indices
            + self.beam_size * torch.arange(bsz, device=scores.device).unsqueeze(1)
        ).view(-1)
        tokens = tokens[sorted_indices]
        scores = scores[sorted_indices]

        # prepare the return value
        retval = {
            "tokens": tokens.view(bsz, self.beam_size, -1),
            "scores": scores.view(bsz, self.beam_size, -1),
        }
        return retval

    @torch.inference_mode()
    def generate_iterator(
        self,
        src_tokens: Tensor,
    ) -> Iterator:
        incremental_states = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {}
        )
        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        max_len = src_len + IMAGE_TOKENS_COUNT
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        # initialize buffers
        tokens = (
            torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(self.pad)
        )

        # first forward through all the fixed tokens with forced decoding we'll
        # need to handle normalization and prep for bookkeeping of incremental
        # decoding
        start_step = src_tokens.shape[1]
        # set all the forced tokens
        tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
        # compute the model predictions
        model_out = self.model.decoder(
            tokens[:, :start_step],
            incremental_state=incremental_states,
        )
        # temperature and normalization
        # convert to float before the temparture divide to ensure good precision.
        # Avoid dividing by 1.0 to prevent unnecessary numerical instability
        # and always log in float
        model_predictions = model_out[0].float()

        # Return First Token
        yield model_predictions[:, -1, :]
        for step in range(start_step, max_len):
            # find our next tokens and record them
            # protect this step for the last token so we don't overflow
            next_toks = self.shared_state[-1].tokens
            tokens[:, step] = next_toks
            # forward through the next pass
            model_out = self.model.decoder(
                tokens[:, : step + 1],
                incremental_state=incremental_states,
            )
            # see above for why this must remain float
            model_predictions = model_out[0].float()[:, -1, :]
            # Return First Token
            yield model_predictions
