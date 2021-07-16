import taichi as ti
from include import *

@ti.data_oriented
class DevMemory:
    def __init__(self):
        capacity = MEM_CAPACITY
        self.capacity = capacity  # max number of particles
        self._size = 0            # cur number of particles
         # PBDynamics
        self.curPos     = self.new_field(capacity) # X, true position
        self.newPos     = self.new_field(capacity) # P, estimated position
        self.velocity   = self.new_field(capacity) # V, velocity
        self.force      = self.new_field(capacity) # F/Acceleration, other than gravity
        self.mass       = self.new_field(capacity, 1) 
        self.phase      = self.new_field(capacity, 1, PHASE_TYPE)
        self.lifetime   = self.new_field(capacity, 1, COUNTER_TYPE)  # [-1]not applicable; [0]dead; [>0]alive

    @staticmethod
    def new_field(shape, dim=2, dtype=ti.f32):
        ''' Allocate Memory on Device '''
        if dim == 2:
            return ti.Vector.field(2, dtype=dtype, shape=shape)
        else:
            return ti.field(dtype, shape=shape)
        
    def clear(self):
        ''' Resetting counters. Old data will be overwritten as the new simulation proceed '''
        self._size  = 0         

    def getNextId(self):
        return self._size

    def size(self):
        return self._size

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
        self._size += 1
