from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension # Or CUDAExtension?? need to check Jonas

setup(
    name='gemmmapreduce_cpp_extensions', # Updated package name for clarity
    version='0.1.1', # Incremented version
    author='Amaru and Jonas', # Optional
    author_email='jonas.lind@ai.se', # Optional
    description='A C++ extension for GeMMMapReduce with a ReLU-MatMul operation adn stuff for Amaru implemenations.', # Optional

    packages=find_packages(where='.', include=['gemmmapreduce*']),
    ext_modules=[
        CppExtension(
            name='relu_matmul_cpp_lib', # Keep the old one if still needed
            sources=['cpp_src/relu_matmul.cpp'],
        ),
        CppExtension( # NEW: Add an extension for custom attention
            name='custom_attention_cpp_lib', # Name of the compiled .so for attention
            sources=['cpp_src/custom_attention.cpp'],
        ),
        # If you were to write custom CUDA kernels in .cu files for attention, check Jonas to learn how to use CUDAExtension:
        # CUDAExtension(
        #     name='custom_attention_cuda_lib',
        #     sources=['cpp_src/custom_attention_wrapper.cpp', 'cpp_src/custom_attention_kernel.cu']
        # )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)