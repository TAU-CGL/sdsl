[![CMake Build (Multi-Platform)](https://github.com/TAU-CGL/sdsl/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/TAU-CGL/sdsl/actions/workflows/cmake-multi-platform.yml) [![Tests](https://github.com/TAU-CGL/sdsl/actions/workflows/python-test.yml/badge.svg)](https://github.com/TAU-CGL/sdsl/actions/workflows/python-test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TAU-CGL/sdsl/blob/main/LICENSE)
-----

![banner](resources/images/sdsl_banner.png?raw=true)

# SDSL: Sparse Distance Sampling Localization

Unified library for all Sparse Distance Sampling Localization methods.
Based on our previous works, presented in ICRA'23, WAFR'24, ICRA'25.
Supports deterministic solution for the kidnapped-robot problem for different kinds of environments and configuration spaces. Also contains implementaion of particle filters for the same problem for comparison.

## Install

### Prerequisites

For macOS (Tested an Apple Sillicon):

    brew install libomp

### Python Bindings

To install the Python bindings, run:

    pip3 install ./sdsl

This package uses `skbuild-conan` to automatically insteall all other necessary dependencies. 