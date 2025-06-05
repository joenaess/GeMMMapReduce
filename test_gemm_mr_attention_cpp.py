import torch
import os
from gemmmapreduce.core import check # Or just check_equality if backward is not yet ready/expected
from gemmmapreduce.attention import regular_attention as python_regular_attention # For comparison

# --- Load Library ---
project_root = os.path.dirname(os.path.abspath(__file__))
so_file_name_pattern = "gemm_map_reduce_attention_lib" # From setup.py
compiled_lib_path = None
for f in os.listdir(project_root):
    if f.startswith(so_file_name_pattern) and f.endswith((".so", ".pyd")):
        compiled_lib_path = os.path.join(project_root, f)
        break

if compiled_lib_path and os.path.exists(compiled_lib_path):
    try:
        torch.ops.load_library(compiled_lib_path)
        print(f"Successfully loaded GeMMMapReduce Attention C++ library: {compiled_lib_path}")
    except Exception as e:
        print(f"Error loading library {compiled_lib_path}: {e}")
        exit()
else:
    print(f"Could not find compiled library {so_file_name_pattern} in {project_root}")
    exit()
# --- End Library Loading ---

def gemm_mr_attention_cpp_wrapper(q, k, v, query_chunk_size=256, kv_chunk_size=256):
    # Our C++ function returns (log_weights, weighted_values)
    # The original gemmmr_attention in attention.py returns only weighted_values after an apply.
    # Let's return the weighted_values for direct comparison with regular_attention
    _, weighted_values = torch.ops.gemm_map_reduce_attention_cpp_ops.forward(
        q, k, v, query_chunk_size, kv_chunk_size
    )
    return weighted_values

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Parameters from attention.py's test
    M, N, D_val_py, F_qk_py = 1024, 1024, 32, 32 
    # Our C++ code uses D_val for value dim, F for QK dim
    # Q (M, F_qk_py), K (N, F_qk_py), V (N, D_val_py)
    # Output (M, D_val_py)

    Q = torch.randn(M, F_qk_py, requires_grad=True, dtype=torch.double, device=device)
    K = torch.randn(N, F_qk_py, requires_grad=True, dtype=torch.double, device=device)
    V = torch.randn(N, D_val_py, requires_grad=True, dtype=torch.double, device=device)

    inputs = (Q, K, V)
    mock_grad = torch.randn(M, D_val_py, dtype=torch.double, device=device)

    # Note: The python_regular_attention from attention.py does NOT include scaling.
    # The C++ proj_fold we wrote also does not include scaling by default to match.
    # If scaling was desired, it should be added consistently.
    
    print("\nTesting GeMMMapReduce Attention C++ vs PyTorch Regular Attention")

    # The `check` function will test forward and backward.
    # PyTorch's autograd will work for the C++ version because it's built from ATen ops.
    # A fully custom C++ backward for the GeMMMapReduce logic would be a next step.
    check(gemm_mr_attention_cpp_wrapper, python_regular_attention, inputs, mock_grad)

    # compare against the original Python GeMMMapReduce attention:
    from gemmmapreduce.attention import gemmmr_attention as python_gemmmr_attention
    print("\nTesting GeMMMapReduce Attention C++ vs Python GeMMMapReduce Attention")
    check(gemm_mr_attention_cpp_wrapper, python_gemmmr_attention, inputs, mock_grad)