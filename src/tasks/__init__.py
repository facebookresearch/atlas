# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import base, fever, kilt, lm, mlm, multiple_choice, qa, section

AVAILABLE_TASKS = {m.__name__.split(".")[-1]: m for m in [base, mlm, lm, multiple_choice, kilt, section, fever, qa]}


def get_task(opt, tokenizer):
    if opt.task not in AVAILABLE_TASKS:
        raise ValueError(f"{opt.task} not recognised")
    task_module = AVAILABLE_TASKS[opt.task]
    return task_module.Task(opt, tokenizer)
