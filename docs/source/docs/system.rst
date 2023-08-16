Initializing a System
======================

The first step in the ``wannierberri`` calculation is initialising the System.  This is done by means of child classes :class:`~wannierberri.__system.System` described below. 
They all have an important common method :func:`~wannierberri.System.set_symmetry`. 
The system may come either from :ref:`Wanier functions <sec-wan-fun>`  constructed by `Wannier90 <http://wannier90.org>`_, or from ref:`tight binding <sec-tb-models>` models. 

.. autoclass:: wannierberri.system.System
   :members: set_parameters, set_symmetry, set_structure, set_symmetry_from_structure
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Symmetrization of the system
-----------------------------

.. automethod:: wannierberri.system.System.symmetrize


.. _sec-wan-fun:

From Wannier functions 
-----------------------------

Wannier90
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_w90
   :show-inheritance:

FPLO
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_fplo
   :show-inheritance:


ASE
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_ASE
   :show-inheritance:



.. _sec-tb-models:


From tight-binding models 
----------------------------------

``wannier90_tb.dat`` file
+++++++++++++++++++++++++

.. autoclass:: wannierberri.system.System_tb
   :show-inheritance:

PythTB
+++++++++

.. autoclass:: wannierberri.system.System_PythTB
   :show-inheritance:

TBmodels
+++++++++

.. autoclass:: wannierberri.system.System_TBmodels
   :show-inheritance:

.. _sec-kp-model:

:math:`\mathbf{k}\cdot\mathbf{p}` models
------------------------------------------

.. autoclass:: wannierberri.system.SystemKP
   :show-inheritance:



