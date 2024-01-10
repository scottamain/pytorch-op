import torch

torch.ops.load_library("build/libaddtensors.so")
print(torch.ops.my_ops.add_tensors(torch.ones((1, 10)), torch.ones((1, 10))))

# model = torch.nn.Sequential(
#   torch.nn.Linear(10, 10),
#   torch.ops.my_ops.add_tensors(),
#   torch.nn.Linear(10, 10)
# )

# model(torch.randn(10, 10))