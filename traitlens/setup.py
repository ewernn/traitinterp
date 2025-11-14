"""
Setup configuration for traitlens package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="traitlens",
    version="0.1.0",
    author="ewernn",
    description="Minimal toolkit for extracting and analyzing trait vectors from transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ewernn/per-token-interp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
        "examples": [
            "transformers>=4.30.0",
        ],
    },
)
