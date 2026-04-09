SDSL — Sparse Distance Sampling Localization
============================================

.. image:: https://github.com/TAU-CGL/sdsl/actions/workflows/cmake-multi-platform.yml/badge.svg
   :target: https://github.com/TAU-CGL/sdsl/actions/workflows/cmake-multi-platform.yml
.. image:: https://github.com/TAU-CGL/sdsl/actions/workflows/python-test.yml/badge.svg
   :target: https://github.com/TAU-CGL/sdsl/actions/workflows/python-test.yml

**SDSL** is a unified C++ library (with Python bindings) for deterministic robot
localization based on sparse distance sampling.  It solves the *kidnapped-robot
problem* — recovering the robot's pose from a handful of distance measurements —
without particle filters or probabilistic priors.

The library is based on research presented at ICRA'23, WAFR'24 and ICRA'25.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started
   tutorials/configuration_space
   tutorials/environments
   tutorials/predicates
   tutorials/localization
   tutorials/python_bindings

.. toctree::
   :maxdepth: 1
   :caption: C++ API Reference

   api/library_root
