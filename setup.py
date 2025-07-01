from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torch_tcu',
      ext_modules=[cpp_extension.CppExtension('torch_tcu', ['torch_tcu.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
