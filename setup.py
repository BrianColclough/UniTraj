import os

from setuptools import find_namespace_packages, setup

# The CUDA extensions are only used by MTR and need a CUDA toolkit that matches the one PyTorch was built with.
# They are opt-in so that the other models install without a compiler:
#   UNITRAJ_BUILD_MTR_OPS=1 pip install -e . --no-build-isolation
BUILD_MTR_OPS = os.environ.get('UNITRAJ_BUILD_MTR_OPS', '0') == '1'


def make_cuda_ext(name, module, sources):
    from torch.utils.cpp_extension import CUDAExtension
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def mtr_ops():
    from torch.utils.cpp_extension import BuildExtension
    return dict(
        cmdclass={
            'build_ext': BuildExtension,
        },

        ext_modules=[
            make_cuda_ext(
                name='knn_cuda',
                module='unitraj.models.mtr.ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_gpu.cu',
                    'src/knn_api.cpp',
                ],
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='unitraj.models.mtr.ops.attention',
                sources=[
                    'src/attention_api.cpp',
                    'src/attention_func_v2.cpp',
                    'src/attention_func.cpp',
                    'src/attention_value_computation_kernel_v2.cu',
                    'src/attention_value_computation_kernel.cu',
                    'src/attention_weight_computation_kernel_v2.cu',
                    'src/attention_weight_computation_kernel.cu',
                ],
            ),
        ],
    )


if __name__ == '__main__':
    setup(
        name='UniTraj',
        version=1.0,
        description='A Unified Framework for Scalable Vehicle Trajectory Prediction',
        author='Lan Feng',
        author_email='fenglan18@outlook.com',
        license='Apache License 2.0',
        # unitraj/ has no __init__.py, so it has to be collected as a namespace package
        packages=find_namespace_packages(include=['unitraj', 'unitraj.*'], exclude=['unitraj.data_samples*']),
        **(mtr_ops() if BUILD_MTR_OPS else {}),
    )
