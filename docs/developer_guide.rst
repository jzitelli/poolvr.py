Developer guide:
++++++++++++++++

``poolvr.table``: Model of the pool table's geometry and other physical parameters
==================================================================================

.. automodule:: poolvr.table
.. autoclass:: poolvr.table.PoolTable
   :members:

``poolvr.physics``: event-based physics engine
==============================================

.. automodule:: poolvr.physics
.. autoclass:: poolvr.physics.PoolPhysics
   :members:

``poolvr.game``: Pool game classes which implement game rules, initial conditions, table setup, etc.
====================================================================================================

.. automodule:: poolvr.game
.. autoclass:: poolvr.game.PoolGame
   :members:

``poolvr.gl_rendering``: OpenGL renderer, node-based scenegraph with glTF-like datatypes
========================================================================================

.. automodule:: poolvr.gl_rendering
.. autoclass:: poolvr.gl_rendering.OpenGLRenderer
   :members:

``poolvr.primitives``: Various primitive geometry classes for the OpenGL renderer and the physics engine
========================================================================================================

.. automodule:: poolvr.primitives
.. autoclass:: poolvr.primitives.HexaPrimitive
   :members:
.. autoclass:: poolvr.primitives.CylinderPrimitive
   :members:

``poolvr.billboard_particles``: OpenGL billboard particle class which is used to render the balls
=================================================================================================



..
   .. toctree::
      physics
      :maxdepth: 2
