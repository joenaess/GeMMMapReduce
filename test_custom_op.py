import torch

# Adjust the path if needed, but with '--inplace' it should be in the current directory
# The exact name comes from your build output:
so_file_name = "relu_matmul_cpp_lib.cpython-310-x86_64-linux-gnu.so" 
try:
    torch.ops.load_library(so_file_name)
    print(f"Successfully loaded {so_file_name}")
except Exception as e:
    print(f"Error loading shared library {so_file_name}: {e}")
    print("Ensure the .so file exists in the current directory or provide the correct path.")
    exit()

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    A = torch.randn(100, 200, dtype=torch.float32, device=device)
    B = torch.randn(200, 300, dtype=torch.float32, device=device)

    try:
        output_custom = torch.ops.relu_matmul_ext.relu_matmul(A, B) # 'relu_matmul_ext' from TORCH_LIBRARY
        print("Custom C++ relu_matmul output shape:", output_custom.shape)
        print("Custom C++ relu_matmul output device:", output_custom.device)

        output_pytorch = torch.relu(torch.matmul(A, B))

        if torch.allclose(output_custom, output_pytorch):
            print("Outputs are close! Custom C++ op matches PyTorch.")
        else:
            print("Outputs differ! There's an issue.")
            print("Custom output:", output_custom)
            print("PyTorch output:", output_pytorch)

    except Exception as e:
        print(f"Error running custom op: {e}")

if __name__ == '__main__':
    main()