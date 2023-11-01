#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq.cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import ast
import random
import sys
import gc
import logging
import functools
import re
import math
from tqdm import tqdm
from rdkit import Chem
import time
import selfies as sf
import parmap
import gradio as gr
import csv

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.sequence_generator_simple import ImageSequenceGenerator
from metaseq.service.utils import build_logger


import importlib

import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb

rkrb.DisableLog("rdApp.error")

logger = build_logger()


def input_loop():
    inp = []
    while True:
        try:
            # green display, bold user prompt
            display = (
                "\033[32mPrompt (ctrl-D to end input, ctrl-C to quit):\n\033[0;1m"
                if not inp
                else ""
            )
            data = input(display)
            inp.append(data)
        except KeyboardInterrupt:
            # reset the formatting
            sys.stdout.write("\033[0m")
            raise
        except EOFError:
            break
        # reset the formatting
        sys.stdout.write("\033[0m")
    logger.debug(f"Input: {inp}")
    return "\n".join(inp)


def selfies_to_smiles(selfies):
    smiles = sf.decoder(selfies)
    return smiles


def worker_main(cfg: MetaseqConfig, namespace_args=None):
    global generator
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))

    generator_old = GeneratorInterface(cfg)
    models = generator_old.load_model()  # noqa: F841
    tgt_dict = generator_old.tgt_dict

    # quiet some of the stuff for visual aspects
    logging.getLogger("metaseq.hub_utils").setLevel(logging.WARNING)

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")

    if torch.distributed.is_initialized():
        request_object = distributed_utils.broadcast_object(
            None, src_rank=0, group=distributed_utils.get_global_group()
        )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        batch_size = cfg.dataset.batch_size
        file_path = cfg.generation.output_file_path

        if os.path.exists(file_path):
            print(f"{file_path} already exists")
            sys.exit(1)

        csv_file = open(file_path, "wt+")
        write_func = functools.partial(csv_file.write)

        # Shape: (batch_size, 2)
        prompt = torch.LongTensor([2, 0]).repeat(batch_size, 1)
        progress = gr.Progress()

        generator = ImageSequenceGenerator(
            models[0],
            tgt_dict,
            progress,
            beam_size=cfg.generation.beam,
            temperature=cfg.generation.temperature,
            topp=cfg.generation.sampling_topp,
        )

        for i in tqdm(range(math.ceil(cfg.generation.generation_len / batch_size))):
            generations = generator.forward(prompt)

            # Clear cuda memory
            torch.cuda.empty_cache()
            gc.collect()

            # Shape: (batch_size, 64)
            generations = generations["tokens"].reshape(-1, 64)
            generations_text = []

            for token_idx_list in generations:
                token_text_list = [
                    generator_old.bpe.bpe.decode([token_idx])
                    for token_idx in token_idx_list
                    if token_idx not in {0, 1, 2, 3}
                ]
                generations_text.append("".join(token_text_list))

            if cfg.generation.mol_repr == "sf":
                # if selfies convert to smiles
                generations_text = parmap.map(
                        selfies_to_smiles, generations_text, pm_processes=2
                    )
                
            # write in a file
            write_func("\n".join(generations_text) + "\n")

        csv_file.close()

        print(f"Saved in {file_path}.")

    else:
        # useful in FSDP setting
        while True:
            request_object = distributed_utils.broadcast_object(
                None, src_rank=0, group=distributed_utils.get_global_group()
            )
            _ = generator.generate(**request_object)


def cli_main():
    """
    Command line interactive.
    """
    parser = options.get_generation_parser()
    args, extra = options.parse_args_and_arch(parser, parse_known=True)
    cfg = convert_namespace_to_omegaconf(args)

    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    if os.getenv("SLURM_NODEID") is None:
        logger.warning(
            f"Missing slurm configuration, defaulting to 'use entire node' for API"
        )
        os.environ["SLURM_NODEID"] = "0"
        os.environ["SLURM_NNODES"] = "1"
        os.environ["SLURM_NTASKS"] = "1"
        import socket

        os.environ["SLURM_STEP_NODELIST"] = socket.gethostname()
    cli_main()
