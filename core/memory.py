import taichi as ti
from include import *

@ti.data_oriented
class DevMemory:
    def __init__(self, capacity = MEM_CAPACITY):
        self.capacity = capacity                            # max number of particles
        self.dev_size = new_field((), 1, COUNTER_TYPE) # cur number of particles(on device)
        self.hst_size = 0                                   # cur number of particles(on host)
        # PBDynamics
        self.curPos     = new_field(capacity) # X, true position
        self.newPos     = new_field(capacity) # P, estimated position
        self.velocity   = new_field(capacity) # V, velocity
        self.force      = new_field(capacity) # F/Acceleration, other than gravity
        self.mass       = new_field(capacity, 1) 
        self.phase      = new_field(capacity, 1, PHASE_TYPE)
        self.lifetime   = new_field(capacity, 1, COUNTER_TYPE)  # [-1]not applicable; [0]dead; [>0]alive
        # Computational Cache
        self.lambdas = new_field(capacity, 1)  # constraint
        self.deltaX  = new_field(capacity)     # position change
    
        
    def clear(self):
        ''' Resetting counters. Old data will be overwritten as the new simulation proceed '''
        self.dev_size[None] = 0         
        self.hst_size       = 0

    def getNextId(self):
        return self.hst_size

    @ti.func
    def size(self):
        return self.dev_size[None]

    def add(self, particle:Particle):
        if DEBUG_MODE:
            assert self.getNextId() == particle.id
        idx = self.getNextId()
        self.curPos[idx]    = particle.pos
        self.velocity[idx]  = particle.vel
        self.force[idx]     = particle.acc
        self.mass[idx]      = particle.mass
        self.phase[idx]     = particle.phase
        self.lifetime[idx]  = particle.lifetime
        # increment counter
        self.dev_size[None] += 1
        self.hst_size       += 1
