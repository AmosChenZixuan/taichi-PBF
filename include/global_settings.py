import taichi as ti
from src.vector import *

# architeture to run on
ARCH = ti.gpu

# if turn on, sanity checks will be performed during simulation
DEBUG_MODE = True

# max number of particles allowed
# TODO: make it dynamic 
MEM_CAPACITY = 2**13

# Types 
PHASE_TYPE   = ti.i8
INDEX_TYPE   = ti.i32
COUNTER_TYPE = ti.i32