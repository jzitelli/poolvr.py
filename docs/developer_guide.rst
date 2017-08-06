Developer guide:
++++++++++++++++


``poolvr.game``: Pool game classes which implement game rules, initial conditions, table setup, etc.
====================================================================================================

.. automodule:: poolvr.game
.. autoclass:: poolvr.game.PoolGame
   :members:


``poolvr.table``: Model of the pool table's geometry and other physical parameters
==================================================================================

.. automodule:: poolvr.table
.. autoclass:: poolvr.table.PoolTable
   :members:


``poolvr.physics``: event-based Pool physics simulator
======================================================

.. automodule:: poolvr.physics
.. autoclass:: poolvr.physics.PoolPhysics
   :members:


``poolvr.ode_physics``: Open Dynamics Engine (ODE)-based physics simulator (time-stepped)
=========================================================================================

.. automodule:: poolvr.ode_physics
.. autoclass:: poolvr.ode_physics.ODEPoolPhysics
   :members:


``poolvr.gl_rendering``: OpenGL renderer, node-based scenegraph with glTF-like datatypes
========================================================================================

.. automodule:: poolvr.gl_rendering
.. autoclass:: poolvr.gl_rendering.OpenGLRenderer
   :members:
.. autoclass:: poolvr.gl_rendering.Program
   :members:
.. autoclass:: poolvr.gl_rendering.Technique
   :members:
.. autoclass:: poolvr.gl_rendering.Texture
   :members:
.. autoclass:: poolvr.gl_rendering.CubeTexture
   :members:
.. autoclass:: poolvr.gl_rendering.Material
   :members:
.. autoclass:: poolvr.gl_rendering.Primitive
   :members:
.. autoclass:: poolvr.gl_rendering.Mesh
   :members:
.. autoclass:: poolvr.gl_rendering.Node
   :members:


``poolvr.primitives``: Various primitive geometry classes for the OpenGL renderer and the physics engine
========================================================================================================

.. automodule:: poolvr.primitives
.. autoclass:: poolvr.primitives.HexaPrimitive
   :members:
.. autoclass:: poolvr.primitives.CylinderPrimitive
   :members:
.. autoclass:: poolvr.primitives.SpherePrimitive
   :members:
.. autoclass:: poolvr.primitives.PlanePrimitive
   :members:
.. autoclass:: poolvr.primitives.BoxPrimitive
   :members:


``poolvr.billboards``: OpenGL billboard particle class which is used to render the balls
========================================================================================

.. automodule:: poolvr.billboards



..
   .. toctree::
      physics
      :maxdepth: 2


``poolvr.app``: Main application
================================

.. automodule:: poolvr.app
   :members:

