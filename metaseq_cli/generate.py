#!/usr/bin/env python3 -u
"""
Generate a sentence by a given prompt.
"""

import argparse
import functools
import logging
import math
import os
import subprocess
import sys
import time
from typing import Dict, Optional, Any, List, Tuple, Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.profiler as profiler
from omegaconf import DictConfig, OmegaConf

from metaseq import (
    checkpoint_utils,
    options,
    tasks,
    utils,
)
from metaseq.data import iterators, data_utils
from metaseq.data.plasma_utils import PlasmaStore
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from metaseq.file_io import PathManager
from metaseq.logging import meters, metrics, progress_bar
from metaseq.model_parallel.megatron_trainer import MegatronTrainer
from metaseq.trainer import Trainer

from metaseq.models.transformer_lm import TransformerLanguageModel
from pathlib import Path
import json

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger("metaseq_cli.train")


def main(cfg: DictConfig) -> None:
    utils.import_user_module(cfg.common)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print nvidia smi stats
    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    model = TransformerLanguageModel.from_pretrained(
        cfg.checkpoint.save_dir,
        checkpoint_file="checkpoint_last.pt",
        bpe="gpt2"
    ).models[0]
    model.eval()
    model = model.to(torch.cuda.current_device())


    # TODO(tmyn) check this
    # model.generate

    criterion = task.build_criterion(cfg.criterion)
    trainer = Trainer(cfg, task, model, criterion)

    tmp_dir = Path("generation-tmp/valid")
    tmp_subset = Path("00")
    tmp_dest = Path(cfg.task.data).joinpath(tmp_dir, tmp_subset)
    os.makedirs(tmp_dest, exist_ok=True)

    generate(cfg, task, model, trainer, tmp_dest, tmp_dir)


def generate(cfg, task, model, trainer, tmp_dest, tmp_dir, prompt=None):
    if not prompt:
        prompt = input("input prompt:")
    sample_dict = {"text": prompt}
    file_name = f"a.jsonl"
    file_dest = tmp_dest.joinpath(file_name)

    with open(file_dest, "w") as f:
        json.dump(sample_dict, f)

    task.load_dataset(tmp_dir)
    itr_valid = trainer.get_valid_iterator(tmp_dir).next_epoch_itr(shuffle=False, set_dataset_epoch=False)

    sample = next(itr_valid)
    sample = _move_to(sample, torch.cuda.current_device())
    output, _ = model(**sample["net_input"])

    current_idx = [i for i, idx in enumerate(sample['net_input']['src_tokens'].squeeze(
        0).tolist()) if idx != task.dictionary.pad()][-1]
    logits = output[:, current_idx, :]

    if cfg.criterion.decoding == "greedy":
        next_token = task.dictionary.symbols[torch.argmax(logits)]
        print(f"generated text: {prompt} {next_token}", end="\r")
        stop_generate = input()
        if not stop_generate:
            generate(cfg, task, model, trainer, tmp_dest, tmp_dir, prompt=f"{prompt} {next_token}")
    else:
        raise NotImplementedError(f"decoding method {cfg.criterion.decoding}")


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def _move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, OrderedDict):
        res = OrderedDict()
        for k, v in obj.items():
            res[k] = _move_to(v, device)
        return res
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = _move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(_move_to(v, device))
        return res
    else:
        return obj


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_generation_parser()
    parser.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        help="Method of sentence generation decoding",
    )
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    # For training - this is where arg parsing happens.
    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
