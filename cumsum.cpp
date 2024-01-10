#include <torch/script.h>

torch::Tensor cumsum(torch::Tensor x, int64_t dim) {
  auto out = torch::empty_like(x);

  float* inPtr = x.data_ptr<float>();
  float* outPtr = out.data_ptr<float>();

  float val = 0.0f;
  for (int64_t i = 0; i < x.size(dim); ++i) {
    val += inPtr[i];
    outPtr[i] = val;
  }

  return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("cumsum", &cumsum);
}