from setuptools import setup, Extension

extension = Extension(
    name='biset', 
    sources=['biset.cpp'],
    extra_compile_args=['-O3']
)

setup(name="biset",
      version="1.0.0",
      description="biset Module",
      ext_modules=[extension])
