[![CMake on multiple platforms](https://github.com/TAU-CGL/sdsl/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/TAU-CGL/sdsl/actions/workflows/cmake-multi-platform.yml)


![banner](resources/images/sdsl_banner.png?raw=true)

# SDSL: Sparse Distance Sampling Localization

Unified library for all Sparse Distance Sampling Localization methods.
Based on our previous works, presented in ICRA'23, WAFR'24, ICRA'25.
Supports deterministic solution for the kidnapped-robot problem for different kinds of environments and configuration spaces. Also contains implementaion of particle filters for the same problem for comparison.

## Install

### Prerequisites

For Ubuntu 20.04 LTS (focal):

    sudo apt-get install libfmt-dev libcgal-dev libglm-dev

For macOS (Tested an Apple Sillicon):

    brew install libomp fmt cgal glm

### Python Bindings

To install the Python bindings, run:

    pip3 install ./sdsl

<!-- ## Usage -->

<!-- See `python/demo.py` for an example. -->

## Data processing

### 2D Data

We process the JackRabbot, DROW and FROG datasets.
TODO. Explain where (from and to) download these datasets - into `data/raw`, which is .gitignored.