#include <torch/torch.h>

torch::Tensor add(torch::Tensor input) {
  return input + 1;
}

// static auto registry = torch::RegisterOperators("my_ops::add", add);

TORCH_LIBRARY(my_ops, m) {
  m.def("add", &add);
}