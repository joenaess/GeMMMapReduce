from setuptools import setup, find_packages
# from torch.utils.cpp_extension import BuildExtension, CppExtension # Or CUDAExtension?? need to check Jonas
# Important: Use CUDAExtension now
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # CUDAExtension for CUDA-specific builds nstead of CppExtension

setup(
    name='gemmmapreduce_cpp_extensions',
    version='0.1.3', 
        author='Amaru and Jonas', # Optional
    author_email='jonas.lind@ai.se', # Optional
    description='A C++ extension for GeMMMapReduce with a ReLU-MatMul operation and stuff for Amaru implemenations. Now with .cu', # Optional
    packages=find_packages(where='.', include=['gemmmapreduce*']),
    ext_modules=[
        CUDAExtension( 
            name='gemm_map_reduce_attention_lib', # Can keep the same library name
            sources=[
                'cpp_src/gemm_map_reduce_attention.cpp', # Main C++ loop logic
                'cpp_src/custom_proj_fold_kernel.cu'     # Your new CUDA kernel file
            ],
            # Optional: Add extra compile args for nvcc if needed
            # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O3', '--use_fast_math']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)