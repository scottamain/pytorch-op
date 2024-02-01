#include <torch/torch.h>

torch::Tensor my_add(torch::Tensor input) {
  return input + 1;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("my_add", &my_add);
}