from typing import NewType
import taichi as ti
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from core.memory import DevMemory
from include import *

@ti.data_oriented
class shapeMatching:
    def __init__(self, memory: DevMemory, alpha):
        self.mem = memory
        self.ptr   = new_field(memory.capacity, 1, INDEX_TYPE) # global index
        self._size = new_field((), 1, COUNTER_TYPE)            # number of particles bounbed by this constraint 
        # cache
        self.CM  = new_field(())
        self.Apq = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())
        self.R   = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())
        # constant
        self.alpha = alpha

    def init(self):
        if self._size[None] > 0:
            self.updateCM()
            self.update_Q()
            self.init_helper()
    
    @ti.kernel
    def init_helper(self):
        for i in range(self.size()):
            x = self.ptr[i]
            self.mem.Q0[x] = self.mem.Q[x]
    
    def solve(self):
        if self._size[None] > 0:
            self.updateCM()
            self.update_cache()
            self.update_delta()

    @ti.kernel
    def updateCM(self):
        mem = self.mem
        cm  = ti.Vector([0., 0.])
        m   = 0.
        for i in range(self.size()):
            x = self.ptr[i]
            if not mem.lifetime[x]: continue
            cm += mem.newPos[x] * mem.mass[x]
            m  += mem.mass[x]
        cm /= m
        self.CM = cm
    
    
    def update_cache(self):
        self.update_Q()
        self.calc_Apq()
        self.calc_R()

    @ti.kernel
    def update_Q(self):
        mem = self.mem
        for i in range(self.size()):
            x = self.ptr[i]
            mem.Q[x] = mem.newPos[x] - self.CM[None]

    @ti.kernel
    def calc_Apq(self):
        mem = self.mem
        _sum = ti.Vector([[0.,0.],[0.,0.]])
        for i in range(self.size()):
            x     = self.ptr[i]
            _sum += mem.Q[x] @ mem.Q0[x].transpose()
        self.Apq[None] = _sum  

    def calc_R(self):
        A = self.Apq[None].value.to_numpy()
        S = sqrtm(A.T@A)
        self.R[None] = ti.Vector(A @ inv(S))

    @ti.kernel
    def update_delta(self):
        mem = self.mem
        for i in range(self.size()):
            x = self.ptr[i]
            p = self.R[None] @ mem.Q0[x] + self.CM[None]
            mem.newPos[x] += (p - mem.newPos[x]) * self.alpha

    def add(self, particle:Particle):
        idx = self._size[None]
        self.ptr[idx] = particle.id
        self._size[None] += 1

    def clear(self):
        self._size[None] = 0

    @ti.func
    def size(self):
        return self._size[None]