Python API Reference
====================

.. note::

   The ``sdsl`` package must be installed (``pip install ./sdsl``) before
   building these docs so that ``autodoc`` can import the compiled extension.

Core
----

.. autofunction:: sdsl.seed

Configurations
--------------

.. autoclass:: sdsl.R2
   :members:
   :special-members: __getitem__, __setitem__, __repr__, __add__, __sub__

.. autoclass:: sdsl.R3
   :members:
   :special-members: __getitem__, __setitem__, __repr__, __add__, __sub__

.. autoclass:: sdsl.R4
   :members:
   :special-members: __getitem__, __setitem__, __repr__, __add__, __sub__

Voxels
------

.. autoclass:: sdsl.Voxel_R2
   :members:

.. autoclass:: sdsl.Voxel_R3
   :members:

.. autoclass:: sdsl.Voxel_R4
   :members:

Environments
------------

.. autoclass:: sdsl.Environment
   :members:

.. autoclass:: sdsl.Env_R2_Arrangement
   :members:
   :show-inheritance:

.. autoclass:: sdsl.Env_2D_PCD
   :members:
   :show-inheritance:

.. autoclass:: sdsl.Env_3D_PCD
   :members:
   :show-inheritance:

Predicates
----------

.. autoclass:: sdsl.Predicate_Fwd2D_Arr
   :members:

.. autoclass:: sdsl.Predicate_AlwaysTrue_3d
   :members:

.. autoclass:: sdsl.Predicate_AlwaysTrue_4d
   :members:

Localization
------------

.. autofunction:: sdsl.localize_omp_forkjoin_3d

.. autofunction:: sdsl.localize_omp_forkjoin_4d

Simulation utilities
--------------------

.. automodule:: sdsl.simulation
   :members:

.. autofunction:: sdsl.loaders.load_poly_as_pcd.load_poly_as_pcd
