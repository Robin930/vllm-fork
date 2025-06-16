import torch
import time

# CPU tensor creation
start = time.time()
torch.tensor([], dtype=torch.int64)
print("CPU tensor:", time.time() - start)

# CUDA tensor creation
start = time.time()
torch.tensor([], device='cuda:0', dtype=torch.int64)
print("First CUDA tensor:", time.time() - start)

# Second CUDA tensor (should be much faster)
start = time.time()
torch.tensor([], device='cuda:0', dtype=torch.int64)
print("Second CUDA tensor:", time.time() - start)
