// cpp_src/attention_utils.h
#pragma once // Ensures this header is included only once

#include <torch/extension.h>

struct AttentionMonoid {
    torch::Tensor log_weights;      // Corresponds to 'z' in Python attention's binary_reduce
    torch::Tensor weighted_values;  // Corresponds to 'v' in Python attention's binary_reduce

    // Default constructor for uninitialized state
    AttentionMonoid() = default;

    // Constructor to initialize with tensors
    AttentionMonoid(torch::Tensor lw, torch::Tensor wv)
        : log_weights(lw), weighted_values(wv) {}
};