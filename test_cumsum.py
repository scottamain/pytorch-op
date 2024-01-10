import torch

torch.ops.load_library("build/libcumsum.so")
print(torch.ops.my_ops.cumsum(torch.ones((1, 10)), 1))