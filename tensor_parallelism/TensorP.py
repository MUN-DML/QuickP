# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import os

import torch
from torch import nn
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import Shard, distribute_tensor

from torch.distributed.device_mesh import init_device_mesh
# Define the module.
m = Model(...)
tp_mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))
m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
big_tensor = torch.randn(100000, 88)
# Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
my_dtensor = distribute_tensor(big_tensor, tp_mesh, [Shard(dim=0)])
