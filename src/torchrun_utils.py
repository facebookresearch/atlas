# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import subprocess
from logging import getLogger
import torch

logger = getLogger()


def init_distributed_mode_torchrun(params):
    """
    Handle single and multi-GPU for singe-node jobs with torchrun.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    For NCCL verbose mode, use:
    os.environ["NCCL_DEBUG"] = "INFO"
    """
    params.local_rank = int(os.environ["LOCAL_RANK"])
    params.node_id = 0
    params.n_nodes = 1
    params.global_rank = int(os.environ["RANK"])
    params.world_size = int(os.environ["WORLD_SIZE"])
    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1
    params.is_distributed = True

    # summary
    PREFIX = "%i - " % params.global_rank

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        # print("Initializing PyTorch distributed ...")
        # Fix for if gloo sockets are inconsistent
        p1 = subprocess.Popen(["ip", "r"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep", "default"], stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()
        gloo_socket_ifname = subprocess.check_output(["awk", "{print $5}"], stdin=p2.stdout).decode("utf-8").strip()
        p2.stdout.close()
        os.environ["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname

        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )

        global GLOO_GROUP

        GLOO_GROUP = torch.distributed.new_group(
            list(range(params.world_size)),
            backend="gloo",
            timeout=datetime.timedelta(0, 600),
        )
