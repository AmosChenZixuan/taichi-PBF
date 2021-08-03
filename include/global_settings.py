import taichi as ti

# architeture to run on
ARCH = ti.gpu

# if turn on, sanity checks will be performed during simulation
DEBUG_MODE = False
DRAW_GRID = True
DRAW_NEIGHBOR = False
DRAW_VEL_FIELD = False

# control
AUTO_EMIT = True

# max number of particles allowed
# TODO: make it dynamic 
MEM_CAPACITY = 2**15

# max number of particles in each cell
NEIGHBOR_CAPACITY = 64
GRID_CAPACITY     = 64

# Types 
PHASE_TYPE   = ti.i8
INDEX_TYPE   = ti.i32
COUNTER_TYPE = ti.i32