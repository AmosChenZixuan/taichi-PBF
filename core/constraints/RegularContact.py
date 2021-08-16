import taichi as ti
from core.memory import DevMemory
from include import *

@ti.data_oriented
class RegularContactSolver:
    def __init__(self, memory: DevMemory, collision_eps):
        self.mem = memory
        self.collision_eps = collision_eps

        # Contact
        self._size  = new_field((), 1, COUNTER_TYPE)
        self.counts = new_field(memory.capacity, 1, COUNTER_TYPE)
        self.pairs  = new_field(memory.capacity*NEIGHBOR_CAPACITY, 2, INDEX_TYPE)  # solid-other contact pairs  


    def solve(self):
        self.project()

    @ti.kernel
    def project(self):
        mem  = self.mem
        if True:
            for i in range(self.size()):
                #print(self.pairs[i])
                x1, x2 = self.pairs[i]
                r    = mem.newPos[x1] - mem.newPos[x2]
                dist = r.norm()
                w    = mem.mass[x1] + mem.mass[x2]
                mag  = dist - self.collision_eps
                if dist > 0. and w > 0. and mag <= 0.:
                    scale = mag / (w / (mem.mass[x1] * mem.mass[x2]))
                    dp    = (scale / dist) * r

                    if mem.mass[x1] > 0:
                        mem.newPos[x1] += -1/mem.mass[x1] * dp  #/ (self.counts[x1]+1) 
                    if mem.mass[x2] > 0: 
                        mem.newPos[x2] +=  1/mem.mass[x2] * dp  #/ (self.counts[x2]+1) 

    @ti.func
    def add(self, x1, x2):
        idx = self._size[None]
        self.pairs[idx]   = x1, x2
        self._size[None] += 1
        self.counts[x1]  += 1
        self.counts[x2]  += 1
        print(x1, x2, self.pairs[idx], self.counts[x1], self.counts[x2])

    @ti.pyfunc
    def clear(self):
        self._size[None] = 0
        for i in self.counts:
            self.counts[i] = 0

    @ti.func
    def size(self):
        return self._size[None]