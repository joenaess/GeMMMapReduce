#include <torch/extension.h>
#include <vector>

// Basic Scaled Dot-Product Attention: softmax( (Q @ K.T) / sqrt(dim_k) ) @ V
// For simplicity, we'll make scaling optional or fixed for now.
// This version uses ATen ops, which will dispatch to CUDA if inputs are on GPU.
torch::Tensor custom_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    bool scale = true) { // Optional scaling factor based on query/key feature dimension

    TORCH_CHECK(query.dim() >= 2, "Query must be at least 2D");
    TORCH_CHECK(key.dim() >= 2, "Key must be at least 2D");
    TORCH_CHECK(value.dim() >= 2, "Value must be at least 2D");
    TORCH_CHECK(query.size(-1) == key.size(-1), "Query and Key feature dimensions must match");
    TORCH_CHECK(key.size(-2) == value.size(-2), "Key and Value sequence lengths must match");
    TORCH_CHECK(query.options().device() == key.options().device() &&
                key.options().device() == value.options().device(),
                "All input tensors must be on the same device");

    // Q @ K.T
    torch::Tensor qk_t = torch::matmul(query, key.transpose(-1, -2));

    // Scaling: / sqrt(dim_k)
    if (scale) {
        double dim_k = static_cast<double>(query.size(-1));
        qk_t = qk_t / std::sqrt(dim_k);
    }

    // Softmax
    torch::Tensor attn_weights = torch::softmax(qk_t, /*dim=*/-1);

    // Softmax output @ V
    torch::Tensor output = torch::matmul(attn_weights, value);

    return output;
}

// Binding code for Python
// The namespace "custom_attention_ops" will be used in Python: torch.ops.custom_attention_ops.forward
TORCH_LIBRARY(custom_attention_ops, m) {
    m.def("forward", custom_attention_forward);
    // If we had a custom backward, you'd register a function that uses torch::autograd::Function here - i.e. todo Jonas
}