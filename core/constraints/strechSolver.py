from typing import NewType
import taichi as ti
import numpy as np
from core.memory import DevMemory
from core.spatialHasing import SpatialHasher
from include import *

@ti.data_oriented
class StrechSolver:
    def __init__(self, memory: DevMemory, x1, x2, d, stiffness=1.):
        self.mem = memory
        self.x1 = x1
        self.x2 = x2
        self.restLen = d
        self.k       = stiffness

    def solve(self):
        self.project()

    @ti.kernel
    def project(self):
        mem = self.mem
        x1, x2 = self.x1, self.x2
        m1, m2 = mem.mass[x1], mem.mass[x2]
        w1 = m1 if m1 == 0. else 1/m1
        w2 = m2 if m2 == 0. else 1/m2
        if not(w1 == w2 == 0.): 
            n = mem.newPos[x1] - mem.newPos[x2]
            d = n.norm()
            # calc position delta
            dp = n.normalized() * (d - self.restLen) / (w1 + w2)
            mem.newPos[x1] -= dp * w1 * self.k
            mem.newPos[x2] += dp * w2 * self.k

    @ti.pyfunc
    def clear(self):
        pass