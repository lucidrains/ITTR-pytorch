from setuptools import setup, find_packages

setup(
  name = 'ITTR-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.4',
  license='MIT',
  description = 'ITTR - Implementation of the Hybrid Perception Block and Dual-Pruned Self-Attention block',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/ITTR-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
