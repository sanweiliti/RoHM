# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

"""
Helpers for distributed training.
"""

import socket

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0

def setup_dist(device=0):
    """
    Setup a distributed process group.
    """
    global used_device
    used_device = device
    if dist.is_initialized():
        return
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device) # f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    # comm = MPI.COMM_WORLD
    # backend = "gloo" if not th.cuda.is_available() else "nccl"

    # if backend == "gloo":
    #     hostname = "localhost"
    # else:
    #     hostname = socket.gethostbyname(socket.getfqdn())
    # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # os.environ["RANK"] = str(comm.rank)
    # os.environ["WORLD_SIZE"] = str(comm.size)

    # port = comm.bcast(_find_free_port(), root=used_device)
    # os.environ["MASTER_PORT"] = str(port)
    # dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if th.cuda.is_available() and used_device>=0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")