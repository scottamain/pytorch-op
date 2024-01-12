import torch
import torch.nn as nn
import torch.jit

torch.ops.load_library("build/libadd.dylib")

def add(input: torch.Tensor) -> torch.Tensor:
    """ This is needed for the onnx register_custom_op_symbolic call. """
    return torch.ops.my_ops.add(input)

class AddModule(nn.Module):
    def forward(self, input):
        return add(input)


input = torch.ones(1, 10)

# Works:
results = torch.ops.my_ops.add(input)
print("op:", results)

# Works:
model = AddModule()
results = model(input)
print("model:", results)

# Does NOT work:
model = torch.nn.Sequential(
  torch.nn.Linear(10, 10),
  AddModule(),  # WORKS FINE IF THIS LINE IS DELETED
  torch.nn.Linear(10, 10)
)
print(model)
results = model(input)
print("sequential:", results)

torch.onnx.register_custom_op_symbolic("my_ops::add", add, 16)
torch.onnx.export(model, input, 'build/model.onnx', input_names=["input"], output_names=["output"])

# scripted_model = torch.jit.script(model)
# scripted_model.save('build/model.torchscript')