# pytorch-op

Basic example of how to build a custom op for a PyTorch model

```
python3 -m pip install torch

mkdir build; cd build

cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
```
