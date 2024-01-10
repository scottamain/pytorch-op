#include <torch/torch.h>

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
  return a + b;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensors", &add_tensors);
}