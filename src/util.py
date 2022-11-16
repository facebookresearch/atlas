# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from src import dist_utils

Number = Union[float, int]

logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return logger


def init_tb_logger(dirname, is_main):

    tb_logger = None
    if is_main:
        try:
            from torch.utils import tensorboard

            tb_logger = tensorboard.SummaryWriter(dirname)
        except:
            logger.warning("Tensorboard is not available.")
    return tb_logger


def cast_to_precision(model, precision):
    if precision == "fp32":
        return model
    elif precision == "fp16":
        model.to(torch.float16)
    elif precision == "bf16":
        model.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported precision {precision}, must be one of fp32, fp16, bf16")
    return model


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup)) + self.ratio

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        return 1.0


class IndexRefreshScheduler(object):
    def __init__(self, format_str: str, freeze_retriever_steps: int, train_retriever: bool):
        """Build an index refresh scheduler

        format_str: string that specifies the schedule.
            has the format: startstep-endstep:refreshrate,startstep-endstep:refreshrate
            e.g. format_str="0-100:10,100-1000000:500" will refresh the index every 10 steps for the first 100 steps
            and then every 500 steps from step 100 to 1M.

            Syntactic Sugar for a fixed schedule: can just pass in a single number
            e.g. format_str="100" will refresh the index every 100 steps

            -1 to never refresh
        )
        """
        self.format_str = format_str
        self.train_retriever = train_retriever
        self.freeze_retriever_steps = freeze_retriever_steps
        self.steps2rates = IndexRefreshScheduler.parse_index_refresh_schedule_string(format_str)

    @classmethod
    def parse_index_refresh_schedule_string(cls, format_str):
        parsed = []
        if format_str == "-1":
            parsed = [(0, 2**32, 2**32)]
        elif format_str.isdigit():
            parsed = [(0, 2**32, int(format_str))]
        else:
            for piece in format_str.split(","):
                startend, rate = piece.split(":")
                start, end = startend.split("-")
                parsed.append((int(start), int(end), int(rate)))
        return parsed

    def is_time_to_refresh(self, step):
        if not (self.train_retriever or step == 0):  # if retriever is not trained only refresh at step 0
            return False
        if not step == 0 and step < self.freeze_retriever_steps:  # freeze first steps
            return False
        for st, en, rate in self.steps2rates:
            if st <= step < en:
                steps_since_refresh_schedule_change = step - st
                return (steps_since_refresh_schedule_change % rate) == 0
        logger.warn(
            "cant calculate refresh rate for this step, I dont have data here"
            " its likely training step is higher than the specificed refresh rate see --index_refresh_rate for help."
        )
        return False


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model):
    from src.AdamWFP32Copy import AdamWFP32Copy

    retr_optimizer = None
    optim_class = AdamWFP32Copy
    optim_args = {"weight_decay": opt.weight_decay, "betas": (0.9, opt.beta2), "eps": opt.epsilon}
    if opt.is_distributed and opt.shard_optim:
        from fairscale.optim.oss import OSS

        optim_args["optim"] = optim_class
        optim_args["force_broadcast_object"] = True
        optim_class = OSS
    optimizer = optim_class(params=model.reader.parameters(), lr=opt.lr, **optim_args)
    if opt.train_retriever:
        retr_optimizer = optim_class(params=model.retriever.parameters(), lr=opt.lr_retriever, **optim_args)

    retr_scheduler = None
    scheduler_args = {"warmup": opt.warmup_steps, "total": opt.total_steps, "ratio": 0.1}
    if opt.scheduler == "linear":
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == "cosine":
        scheduler_class = CosineScheduler
    elif opt.scheduler == "fixed":
        scheduler_class = FixedScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    if opt.train_retriever:
        retr_scheduler = scheduler_class(retr_optimizer, **scheduler_args)

    return optimizer, scheduler, retr_optimizer, retr_scheduler


def compute_grad_stats(model):
    with torch.no_grad():
        stats = []
        for name, p in get_unwrapped_model_if_wrapped(model).reader.named_parameters():
            if p.grad is not None:
                s1 = torch.min(torch.abs(p.grad)).item()
                s2 = torch.max(torch.abs(p.grad)).item()
                s3 = torch.mean(torch.abs(p.grad)).item()
                s4 = torch.linalg.norm(p.grad).item()
                stats += [s1, s2, s3, s4]
            else:
                stats += [0.0, 0.0, 0.0, 0.0]
        stats = torch.Tensor(stats).cuda()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        stats = stats.view(-1, 4)

        res = {}
        res["skip_example"] = (torch.any(torch.isinf(stats)) or torch.any(torch.isnan(stats))).item()
        res["min"] = stats.min(0)[0][0].item()
        res["max"] = stats.max(0)[0][1].item()
        res["mean"] = stats.mean(0)[2].item()
        return res


def write_output(glob_path, output_path):
    files = list(glob_path.glob("*.txt"))
    files.sort()
    with open(output_path, "w") as outfile:
        for path in files:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, dataset_name, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / "tmp_dir"
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f"{opt.global_rank}.json"
    with open(tmp_path, "w") as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / f"{dataset_name}.jsonl"
        logger.info(f"Writing dataset with scores at {final_path}")
        results_path = list(write_path.glob("*.json"))
        results_path.sort()

        alldata = []
        for path in results_path:
            with open(path, "r") as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, "w") as fout:
            for ex in alldata:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        write_path.rmdir()


def avg_dist_dict(keys, dictionary):
    avg = {}
    for m in keys:
        v = dictionary[m]
        if len(v) > 0:
            avg[m] = np.mean(v)
        else:
            avg[m] = 0.0
        avg[m] = dist_utils.weighted_average(avg[m], len(v))[0]
    return avg


class WeightedAvgStats:
    """provides an average over a bunch of stats"""

    def __init__(self):
        self.raw_stats: Dict[str, float] = defaultdict(float)
        self.total_weights: Dict[str, float] = defaultdict(float)

    def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
        for key, (value, weight) in vals.items():
            self.raw_stats[key] += value * weight
            self.total_weights[key] += weight

    @property
    def stats(self) -> Dict[str, float]:
        return {x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()}

    @property
    def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
        return {x: (self.raw_stats[x] / self.total_weights[x], self.total_weights[x]) for x in self.raw_stats.keys()}

    def reset(self) -> None:
        self.raw_stats = defaultdict(float)
        self.total_weights = defaultdict(float)

    @property
    def average_stats(self) -> Dict[str, float]:
        keys = sorted(self.raw_stats.keys())
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(keys, src=0)
        global_dict = {}
        for k in keys:
            if not k in self.total_weights:
                v = 0.0
            else:
                v = self.raw_stats[k] / self.total_weights[k]
            v, _ = dist_utils.weighted_average(v, self.total_weights[k])
            global_dict[k] = v
        return global_dict


def get_unwrapped_model_if_wrapped(model):
    if hasattr(model, "module"):
        return model.module
    return model
