Undocumented Python objects
===========================

poolvr.billboards
-----------------
Classes:
 * BillboardParticles

poolvr.gl_rendering
-------------------
Functions:
 * calc_projection_matrix
 * set_matrix_from_quaternion
 * set_quaternion_from_matrix

Classes:
 * CubeTexture
 * InvalidPrimitiveException
 * InvalidProgramException
 * InvalidTechniqueException
 * Material -- missing methods:

   - init_gl
   - release
   - use
 * Mesh -- missing methods:

   - draw
   - init_gl
 * OpenGLRenderer -- missing methods:

   - init_gl
   - process_input
   - shutdown
   - update_projection_matrix
 * Primitive -- missing methods:

   - init_gl
 * Program -- missing methods:

   - init_gl
   - release
   - use
 * Technique -- missing methods:

   - init_gl
   - release
   - use
 * Texture

poolvr.ode_physics
------------------
Classes:
 * ODEPoolPhysics -- missing methods:

   - add_cue
   - eval_angular_velocities
   - eval_positions
   - eval_quaternions
   - eval_velocities
   - next_turn_time
   - reset
   - set_cue_ball_collision_callback
   - step
   - strike_ball

poolvr.physics
--------------
Classes:
 * PoolPhysics -- missing methods:

   - set_cue_ball_collision_callback
   - step

poolvr.primitives
-----------------
Functions:
 * triangulate_quad

Classes:
 * CirclePrimitive
 * ConeMesh
 * ConePrimitive
 * HexaPrimitive -- missing methods:

   - init_gl
 * QuadPrimitive
 * RoundedRectanglePrimitive

poolvr.table
------------
Classes:
 * PoolTable -- missing methods:

   - setup_balls

