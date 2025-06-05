// cpp_src/gemm_map_reduce_attention.cpp
#include "attention_utils.h" 
#include <vector>
#include <iostream> 
#include <limits>   
#include <algorithm> // For std::min

// In gemm_map_reduce_attention.cpp
// Declaration of the function from custom_proj_fold_kernel.cu
void proj_fold_cuda_dispatcher(
    const torch::Tensor& query_chunk,
    const torch::Tensor& key_chunk,
    const torch::Tensor& value_chunk,
    torch::Tensor& log_weights_out,    // Output tensor
    torch::Tensor& weighted_values_out // Output tensor
);

// Corresponds to init() in attention.py
AttentionMonoid init_attention_accumulator_cpp(const torch::Tensor& query, const torch::Tensor& value) {
    // query: Full M x F tensor (used for shape and device)
    // value: Full N x D tensor (used for shape and device)
    // M = query.size(0), F = query.size(1)
    // N = value.size(0), D = value.size(1)
    // borde stämma...
    
    int64_t M = query.size(0);
    int64_t D_val = value.size(1); // Dimension of value vectors

    auto options = query.options(); // Get dtype and device from query tensor

    torch::Tensor log_weights_acc = torch::full({M}, -std::numeric_limits<double>::infinity(), options);
    torch::Tensor weighted_values_acc = torch::zeros({M, D_val}, options);

    return AttentionMonoid(log_weights_acc, weighted_values_acc);
}


// Corresponds to proj_fold() in attention.py
// Takes CHUNKS of query, key, value
AttentionMonoid proj_fold_attention_cpp(
    const torch::Tensor& query_chunk,   // M_chunk x F
    const torch::Tensor& key_chunk,     // N_chunk x F
    const torch::Tensor& value_chunk) { // N_chunk x D_val

    // zs = query_chunk @ key_chunk.T ( M_chunk x N_chunk )
    torch::Tensor zs = torch::matmul(query_chunk, key_chunk.transpose(-1, -2));
    
    // z_local = torch.logsumexp(zs, dim=1) ( M_chunk )
    torch::Tensor z_local = torch::logsumexp(zs, /*dim=*/1, /*keepdim=*/false);
    
    // v_local = (zs - z_local[:, None]).exp() @ value_chunk
    // (zs - z_local.unsqueeze(1)) -> M_chunk x N_chunk
    // .exp()                     -> M_chunk x N_chunk (these are softmax weights for the chunk)
    // @ value_chunk (N_chunk x D_val) -> M_chunk x D_val
    torch::Tensor v_local = torch::matmul(torch::exp(zs - z_local.unsqueeze(1)), value_chunk);
    
    return AttentionMonoid(z_local, v_local);
}


// Corresponds to binary_reduce() in attention.py
// Takes:
//   acc_current_val: The current accumulated monoid value for a given query slice (M_chunk dimensions)
//   local_projection_val: The new monoid value projected from a key/value chunk (M_chunk dimensions)
AttentionMonoid binary_reduce_attention_cpp(
    const AttentionMonoid& acc_current_val, 
    const AttentionMonoid& local_projection_val) {

    torch::Tensor a_z = acc_current_val.log_weights;           // M_chunk
    torch::Tensor a_v = acc_current_val.weighted_values;       // M_chunk x D_val
    torch::Tensor b_z = local_projection_val.log_weights;      // M_chunk
    torch::Tensor b_v = local_projection_val.weighted_values;  // M_chunk x D_val

    // z_new = torch.logaddexp(a_z, b_z) ( M_chunk )
    torch::Tensor z_new = torch::logaddexp(a_z, b_z);
    
    // v_new = a_v * torch.exp(a_z - z_new)[:, None] + b_v * torch.exp(b_z - z_new)[:, None]
    // (a_z - z_new).unsqueeze(1) -> M_chunk x 1
    // torch.exp(...) -> M_chunk x 1 (weights for a_v)
    // a_v * exp(...) -> M_chunk x D_val
    torch::Tensor factor_a = torch::exp(a_z - z_new).unsqueeze(1);
    torch::Tensor factor_b = torch::exp(b_z - z_new).unsqueeze(1);
    torch::Tensor v_new = a_v * factor_a + b_v * factor_b;
    
    return AttentionMonoid(z_new, v_new);
}

// Main function to be called from Python
// Returns a tuple of tensors: (final_log_weights, final_weighted_values)
std::tuple<torch::Tensor, torch::Tensor> gemm_map_reduce_attention_forward_cpp(
    torch::Tensor Q_full,         // Full M x F query tensor
    torch::Tensor K_full,         // Full N x F key tensor
    torch::Tensor V_full,         // Full N x D_val value tensor
    int64_t query_chunk_size,   // e.g., 256 from Python attention.py
    int64_t kv_chunk_size) {    // e.g., 256 from Python attention.py

    TORCH_CHECK(Q_full.dim() == 2, "Q must be 2D");
    TORCH_CHECK(K_full.dim() == 2, "K must be 2D");
    TORCH_CHECK(V_full.dim() == 2, "V must be 2D");
    // Add more shape/device checks as needed

    int64_t M_full = Q_full.size(0);
    int64_t N_full = K_full.size(0);

    // Initialize the global accumulator A
    AttentionMonoid A_global = init_attention_accumulator_cpp(Q_full, V_full);

    // Loop over query chunks (mslices)
    for (int64_t m_start = 0; m_start < M_full; m_start += query_chunk_size) {
        int64_t m_end = std::min(m_start + query_chunk_size, M_full);
        torch::Tensor query_chunk = Q_full.slice(/*dim=*/0, m_start, m_end);

        torch::Tensor A_global_log_weights_slice_view = A_global.log_weights.slice(/*dim=*/0, m_start, m_end);
        torch::Tensor A_global_weighted_values_slice_view = A_global.weighted_values.slice(/*dim=*/0, m_start, m_end);
        
        AttentionMonoid current_acc_for_m_slice(
            A_global_log_weights_slice_view.clone(), // Clone for iterative update within n_loop
            A_global_weighted_values_slice_view.clone()
        );

        for (int64_t n_start = 0; n_start < N_full; n_start += kv_chunk_size) {
            int64_t n_end = std::min(n_start + kv_chunk_size, N_full);
            
            // Define key_chunk and value_chunk here
            torch::Tensor key_chunk = K_full.slice(/*dim=*/0, n_start, n_end);            
            torch::Tensor value_chunk = V_full.slice(/*dim=*/0, n_start, n_end);

            // --- This section changes ---
            // 1. Pre-allocate output tensors for the custom CUDA kernel for proj_fold
            //    Ensure to use the c10::IntArrayRef fix if directly using torch::empty with {}
            auto M_chunk_size = query_chunk.size(0);
            auto D_val_dim = V_full.size(1);
            auto common_options = query_chunk.options(); // Get dtype, device

            // For log_weights (shape: M_chunk)
            torch::Tensor local_proj_log_weights = torch::empty(c10::IntArrayRef({M_chunk_size}), common_options);

            // For weighted_values (shape: M_chunk x D_val_dim)
            torch::Tensor local_proj_weighted_values = torch::empty(c10::IntArrayRef({M_chunk_size, D_val_dim}), common_options);

            // 2. Call the dispatcher for your custom CUDA kernel
            proj_fold_cuda_dispatcher(
                query_chunk, key_chunk, value_chunk,
                local_proj_log_weights, local_proj_weighted_values
            );

            AttentionMonoid local_projection(local_proj_log_weights, local_proj_weighted_values);

            current_acc_for_m_slice = binary_reduce_attention_cpp(current_acc_for_m_slice, local_projection);
        }
        
        A_global_log_weights_slice_view.copy_(current_acc_for_m_slice.log_weights);
        A_global_weighted_values_slice_view.copy_(current_acc_for_m_slice.weighted_values);
    }

    return std::make_tuple(A_global.log_weights, A_global.weighted_values);
}

TORCH_LIBRARY(gemm_map_reduce_attention_cpp_ops, m) {
    m.def("forward", gemm_map_reduce_attention_forward_cpp);
}