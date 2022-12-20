from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='cowa3d',
        version='0.1.0',
        description=('cowa 3d perception'),
        # long_description=readme(),
        long_description_content_type='text/markdown',
        author='zhouyue&zhanggefan',
        author_email='lizaozhouke@sjtu.edu.cn',
        keywords='computer vision, 3D object detection',
        packages=['cowa3d_common.' + m for m in find_packages('cowa3d_common')],
        include_package_data=True,
        license='Apache License 2.0',
        ext_modules=[
            make_cuda_ext(
                name='voxel_utils',
                module='cowa3d_common.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cuda.cu'
                ]),
            make_cuda_ext(
                name='eval_utils',
                module='cowa3d_common.ops.eval',
                sources=[
                    'eval_utils.cpp',
                    'matcher.cpp',
                    'affinity.cpp'
                ]),
            make_cuda_ext(
                name='ccl_utils',
                module='cowa3d_common.ops.ccl',
                sources=[
                    'ccl.cpp',
                    'ccl_cuda.cu',
                    'spccl_cuda.cu',
                    'voxel_spccl_cuda.cu',
                    'voxelized_sampling.cu',
                    'sample_gpu.cu'
                ])
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
