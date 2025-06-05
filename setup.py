from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension # Or CUDAExtension?? need to check Jonas

setup(
    name='gemmmapreduce_cpp_extensions', # Updated package name for clarity
    version='0.1.2', # Incremented version
    author='Amaru and Jonas', # Optional
    author_email='jonas.lind@ai.se', # Optional
    description='A C++ extension for GeMMMapReduce with a ReLU-MatMul operation adn stuff for Amaru implemenations.', # Optional
    packages=find_packages(where='.', include=['gemmmapreduce*']),
    ext_modules=[
        CppExtension(
            name='relu_matmul_cpp_lib', 
            sources=['cpp_src/relu_matmul.cpp'],
        ),
        CppExtension( 
            name='custom_attention_cpp_lib', 
            sources=['cpp_src/custom_attention.cpp'], # Assuming this is direct attention
        ),
        CppExtension( # NEW: For the C++ GeMMMapReduce Attention
            name='gemm_map_reduce_attention_lib', 
            sources=['cpp_src/gemm_map_reduce_attention.cpp'], 
            # If attention_utils.h is used by gemm_map_reduce_attention.cpp and not just a struct def,
            # ensure compiler can find it (usually fine if in same dir or add include_dirs)
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)