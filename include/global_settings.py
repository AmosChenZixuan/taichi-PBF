import taichi as ti

# architeture to run on
ARCH = ti.gpu

# if turn on, sanity checks will be performed during simulation
DEBUG_MODE = True
DRAW_GRID = True
DRAW_NEIGHBOR = True

# max number of particles allowed
# TODO: make it dynamic 
MEM_CAPACITY = 2**13

# Types 
PHASE_TYPE   = ti.i8
INDEX_TYPE   = ti.i32
COUNTER_TYPE = ti.i32