import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. PyTorch cannot detect your GPUs.")

# Try to allocate a tensor on GPU to be sure
try:
    if torch.cuda.is_available():
        tensor = torch.tensor([1.0, 2.0]).to('cuda')
        print("Successfully created a tensor on CUDA:", tensor.device)
    else:
        print("Skipping CUDA tensor test as CUDA is not available.")
except Exception as e:
    print(f"Error during CUDA tensor test: {e}")