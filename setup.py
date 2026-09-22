from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='vigs_backends',
    ext_modules=[
        CUDAExtension('vigs_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/vigs.cpp',
                'src/vigs_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',
                    '-gencode=arch=compute_120,code=sm_120',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)

# lietorch_5090 modified upon original lietorch (https://github.com/princeton-vl/lietorch), to support CUDA 12.8/RTX 5090/PyTorch 2.8 environment, if you want to use older environment, please use original lietorch
setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch_5090'},
    ext_modules=[
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch_5090/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'thirdparty/lietorch_5090/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch_5090/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch_5090/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',
                    '-gencode=arch=compute_120,code=sm_120',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
