from skbuild_conan import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "sdsl",
    version = "1.0.0",
    description = "Unified library for all Sparse Distance Sampling Localization methods.",
    readme = "README.md",
    authors = [
        { "name": "Michael Bilevich", "email": "michaelmoshe@mail.tau.ac.il" },
    ],
    python_requires = ">=3.10",
    classifiers = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    conanfile = "./conanfile.txt",
    install_requires=required,
)