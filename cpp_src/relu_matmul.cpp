#include <torch/extension.h> // Main PyTorch C++ extension header
#include <iostream>

// Declare the CUDA forward pass function if you were to write a custom CUDA kernel
// For this example, we'll use ATen ops that dispatch to CUDA, so a separate .cu is not strictly needed yet.
// torch::Tensor relu_matmul_cuda_forward(torch::Tensor a, torch::Tensor b);

// C++ function that will be callable from Python
torch::Tensor relu_matmul_cpu_or_gpu(torch::Tensor a, torch::Tensor b) {
    // Basic input validation (optional, but good practice)
    TORCH_CHECK(a.dim() == 2, "Input tensor 'a' must be 2-dimensional");
    TORCH_CHECK(b.dim() == 2, "Input tensor 'b' must be 2-dimensional");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions of a and b must match for matmul");
    TORCH_CHECK(a.options().device() == b.options().device(), "Tensors 'a' and 'b' must be on the same device");

    // Perform the operations using PyTorch's ATen library
    // These operations will automatically run on the GPU if 'a' and 'b' are CUDA tensors.
    torch::Tensor output = torch::relu(torch::matmul(a, b));
    
    return output;
}

// This is the binding code that makes C++ functions available in Python.
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def(
//    "relu_matmul",                                        옆 Function name in Python
//    &relu_matmul_cpu_or_gpu,                              옆 Pointer to the C++ function
//    "Computes relu(A @ B) using PyTorch ATen library. " 옆 Optional docstring
//    "Dispatches to CUDA if input tensors are on GPU."
//  );
//}

// Updated way to bind for PyTorch 1.5+ (aligns better with torch.hub and JIT)
TORCH_LIBRARY(relu_matmul_ext, m) { // relu_matmul_ext will be part of the module name
    m.def("relu_matmul", relu_matmul_cpu_or_gpu);
}