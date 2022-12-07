# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List
import argparse
import numpy as np
import torch
import torch.cuda
import sys
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from train import train

os.environ["TOKENIZERS_PARALLELISM"] = "true"
NCONTEXT: str = "40"
PBSZ: str = "2"
PRECISION: str = "bf16"
GOLD_SCORE_MODE: str = "ppmean"
GPU_MAX_LENGTH: str = "384"
GEN_MAX_LENGTH: str = "32"
EPSILON: str = "0.01"
SMALL_EPSILON: str = "4e-5"
DROPOUT: str = "0.1"
NO_WARMUP_STEPS: str = "0"
NO_REFRESH: str = "-1"


def set_parser_options(parser: argparse.Namespace, passed_args: List[str]) -> argparse.ArgumentParser:
    """
    Sets some default options for finetuning an Atlas model for a q&a task.
    """

    all_args = [
        "--train_retriever",
        "--query_side_retriever_training",
        "--use_gradient_checkpoint_reader",
        "--use_gradient_checkpoint_retriever",
        "--shard_optim",
        "--shard_grads",
        "--temperature_gold",
        EPSILON,
        "--temperature_score",
        EPSILON,
        "--refresh_index",
        "-1",
        "--dropout",
        DROPOUT,
        "--lr",
        SMALL_EPSILON,
        "--lr_retriever",
        SMALL_EPSILON,
        "--scheduler",
        "linear",
        "--weight_decay",
        EPSILON,
        "--generation_max_length",
        GEN_MAX_LENGTH,
        "--target_maxlength",
        GEN_MAX_LENGTH,
        "--gold_score_mode",
        GOLD_SCORE_MODE,
        "--precision",
        PRECISION,
        "--text_maxlength",
        GPU_MAX_LENGTH,
        "--per_gpu_batch_size",
        PBSZ,
        "--n_context",
        NCONTEXT,
        "--retriever_n_context",
        NCONTEXT,
        "--task",
        "qa",
        "--refresh_index",
        NO_REFRESH,
        "--warmup_steps",
        NO_WARMUP_STEPS,
    ] + passed_args

    return parser.parse_args(all_args)


if __name__ == "__main__":
    options = get_options()
    opt = set_parser_options(options.parser, sys.argv[1:])

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step = load_or_initialize_atlas_model(opt)

    if opt.is_distributed:
        if opt.shard_grads:
            import fairscale.nn.data_parallel

            model.reader = fairscale.nn.data_parallel.ShardedDataParallel(
                model.reader, optimizer, auto_refresh_trainable=False
            )
            if opt.train_retriever:
                model.retriever = fairscale.nn.data_parallel.ShardedDataParallel(
                    model.retriever, retr_optimizer, auto_refresh_trainable=False
                )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            model._set_static_graph()

    logger.info("Start finetuning")
    dist_utils.barrier()
    train(
        model,
        index,
        passages,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        step,
        opt,
        checkpoint_path,
    )
