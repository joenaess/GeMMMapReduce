import torch
import os # For robust path joining
from gemmmapreduce.core import check

# --- Load the custom C++ library for attention ---
# e.g., custom_attention_cpp_lib.cpython-310-x86_64-linux-gnu.so

# Construct path to the .so file (assuming it's in the project root after --inplace)
project_root = os.path.dirname(os.path.abspath(__file__)) # Gets directory of test_custom_attention.py
# You might need to manually find the exact .so file name in your project root after building.
so_file_name_pattern = "custom_attention_cpp_lib" # Base name from setup.py
compiled_lib_path = None
for f in os.listdir(project_root):
    if f.startswith(so_file_name_pattern) and f.endswith((".so", ".pyd")): # .pyd for Windows which nobody should use though...
        compiled_lib_path = os.path.join(project_root, f)
        break

if compiled_lib_path and os.path.exists(compiled_lib_path):
    try:
        torch.ops.load_library(compiled_lib_path)
        print(f"Successfully loaded custom attention library: {compiled_lib_path}")
    except Exception as e:
        print(f"Error loading custom attention library {compiled_lib_path}: {e}")
        exit()
else:
    print(f"Could not find compiled library starting with {so_file_name_pattern} in {project_root}")
    exit()
# --- End Library Loading ---


# Wrapper for the custom C++ attention forward pass
def cpp_custom_attention(q, k, v, scale_cpp=True):
    # The namespace 'custom_attention_ops' and function 'forward'
    # come from TORCH_LIBRARY(custom_attention_ops, m) and m.def("forward", ...)
    return torch.ops.custom_attention_ops.forward(q, k, v, scale_cpp)

# Regular PyTorch attention for comparison (from attention.py)
# Note: The original regular_attention in attention.py doesn't include scaling by default.
# We'll add scaling here for a fair comparison if my C++ version uses it.
def regular_attention_pytorch(q, k, v, scale_pytorch=True):
    scores = torch.matmul(q, k.transpose(-1, -2))
    if scale_pytorch:
        scores = scores / (k.size(-1) ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Parameters (e.g., from attention.py)
    # Batch size, sequence length, feature dimension, value dimension
    B, M, N, F, D_val = 4, 32, 32, 16, 24 # Example dimensions: Batch, SeqLen_Q, SeqLen_KV, FeatDim_QK, FeatDim_V

    # Using a common feature dimension for Q, K
    # Using a common sequence length for K, V
    
    # Make dimensions more explicit for clarity:
    # B: Batch size
    # M_seq_len: Sequence length for Query
    # N_seq_len: Sequence length for Key/Value (can be same or different from M_seq_len)
    # F_qk_dim: Feature dimension for Query/Key
    # D_v_dim: Feature dimension for Value (can be same or different from F_qk_dim)

    # For simplicity, let Q,K,V have same batch, seq_len, and Q,K have same feat_dim
    # B, S, F_qk, F_v = 4, 32, 16, 24 # Batch, SeqLen, Feat_QK, Feat_V
    # Q: (B, S, F_qk)
    # K: (B, S, F_qk)
    # V: (B, S, F_v)
    # Output: (B, S, F_v)
    # For the check function in core.py, it might expect 2D or 3D inputs based on original examples.
    # Let's use 3D inputs (Batch, Seq, Dim) as common in attention.
    # Q (batch, query_seq_len, qk_dim)
    # K (batch, kv_seq_len, qk_dim)
    # V (batch, kv_seq_len, v_dim)
    
    batch_size = 2
    query_seq_len = 10
    kv_seq_len = 12
    qk_dim = 8
    v_dim = 16


    Q = torch.randn(batch_size, query_seq_len, qk_dim, requires_grad=True, dtype=torch.double, device=device)
    K = torch.randn(batch_size, kv_seq_len, qk_dim, requires_grad=True, dtype=torch.double, device=device)
    V = torch.randn(batch_size, kv_seq_len, v_dim, requires_grad=True, dtype=torch.double, device=device)

    inputs = (Q, K, V)
    # Mock gradient should match the output shape: (batch_size, query_seq_len, v_dim)
    mock_grad = torch.randn(batch_size, query_seq_len, v_dim, dtype=torch.double, device=device)
    
    # Set scale_factor to True for both if you want to compare scaled attention
    scale_attention = True

    print(f"\nTesting Custom C++ Attention (scaled={scale_attention}) vs PyTorch Attention (scaled={scale_attention})")
    
    # Need to wrap them if they take extra args not in 'inputs'
    def cpp_att_wrapper(q,k,v):
        return cpp_custom_attention(q,k,v, scale_cpp=scale_attention)

    def regular_att_wrapper(q,k,v):
        return regular_attention_pytorch(q,k,v, scale_pytorch=scale_attention)

    # The `check` function from core.py tests both forward and backward passes.
    # Since our C++ custom_attention_forward uses ATen ops, PyTorch's autograd
    # will be able to compute gradients for it automatically.
    # A fully custom backward pass would require implementing torch::autograd::Function in C++.
    check(cpp_att_wrapper, regular_att_wrapper, inputs, mock_grad)