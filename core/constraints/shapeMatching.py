import taichi as ti
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from core.memory import DevMemory
from include import *

@ti.data_oriented
class shapeMatching:
    def __init__(self, memory: DevMemory):
        self.mem = memory
        # self.ptr   = new_field(memory.capacity, 1, INDEX_TYPE) # global index
        # self._size = new_field((), 1, COUNTER_TYPE)            # number of particles bounbed by this constraint 

        # self.CM  = new_field(())
        # self.CM0 = new_field(())
        # self.Q   = None
        # self.Q0  = None
        # # 
        # self.Apq = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())
        # self.R   = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())

        # constant
        self.alpha = 1/50
    
    def solve(self):
        pass

    def init_memory(self):
        self.Q   = new_field(self.size())
        self.Q0  = new_field(self.size())

    def add(self, particle:Particle):
        idx = self._size[None]
        self.ptr[idx] = particle.id
        self._size[None] += 1

    def clear(self):
        self._size[None] = 0

    @ti.func
    def size(self):
        return self._size[None]