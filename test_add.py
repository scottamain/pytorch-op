import torch
import torch.nn as nn
import torch.jit

torch.ops.load_library("build/libadd.dylib")

class AddModule(nn.Module):
    def forward(self, input):
        return torch.ops.my_ops.add(input)

# Test the op in different use cases
input = torch.ones(1, 10)
print("op:", torch.ops.my_ops.add(input))

model = AddModule()
print("model:", model(input))

model = torch.nn.Sequential(
  torch.nn.Linear(10, 10),
  AddModule(),
  torch.nn.Linear(10, 10)
)
print("sequential:", model(input))

# Now convert to ONNX
def sym_add(g, input):
    return g.op("my_ops::add", input).setType(input.type())

torch.onnx.register_custom_op_symbolic("my_ops::add", sym_add, 1)
torch.onnx.export(model, input, 'build/model.onnx',
                  input_names=["input"], output_names=["output"])

# And TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('build/model.torchscript')