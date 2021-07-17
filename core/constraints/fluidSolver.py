import taichi as ti
import numpy as np
from core.memory import DevMemory
from core.spatialHasing import SpatialHasher
from include import *

@ti.data_oriented
class fluidSolver:
    def __init__(self, memory: DevMemory, grid:SpatialHasher):
        self.mem   = memory
        self.grid  = grid
        self.ptr   = new_field(memory.capacity, 1, INDEX_TYPE) # global index
        self._size = new_field((), 1, COUNTER_TYPE)            # number of particles bounbed by this constraint 
        # const
        self.kernel_size = 25                   # h value for kernels
        self.kernel2     = self.kernel_size**2
        self.poly6_const = 315 / 64 / np.pi / self.kernel_size**9
        self.spikyG_const= -45 / np.pi / self.kernel_size**6
        self.restDensity = (self.poly6_const * self.kernel2**3) * 0.5 # rho_0 = restDensity * mass
        self.relaxation  = 200                  # applied to lambda 
        # Tensile Instability  (repulsive term S_corr)
        deltaQ = 0.3 * self.kernel_size         # 0.1h ~ 0.3h
        self.s_corr_k    = 0.1                  # s_corr = k*(w_ij/wDeltaQ)**n
        self.s_corr_n    = 4
        self.s_corr_const= 1 / (self.poly6_const * (self.kernel2 - deltaQ**2) ** 3) # wploy6(deltaQ)
        # Computational Cache
        self.lambdas = new_field(memory.capacity, 1)  # constraint
        self.deltaX  = new_field(memory.capacity)     # position change
        
    def solve(self):
        self.calc_lambda()
        self.calc_delta()
        self.project()

    @ti.kernel
    def calc_lambda(self):
        '''
        i     : local index; ith particle bounded by this constraint
        x1, x2: global index in memory

        Ci = SUM_j{ wPoly6_ij * m_j}  / rho0  - 1

        if k == i:
            Grad_P_k_Ci = SUM_j{ wSpikyG_ij }
        elif k == j:
            Grad_P_k_Ci = -wSpikyG_ij
        lambda_i = -Ci / (SUM_k{ ||Grad_P_k_Ci|| })
        '''
        mem  = self.mem
        grid = self.grid
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            # accumulators
            rho_i              = self.wPoly6(0.)
            sum_Ci             = vec2()
            sum_grad_pk_Ci_sqr = 0.
            for j in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, j]
                if mem.phase[x2] == SMOKE: continue 
                r    = mem.newPos[x1] - mem.newPos[x2]
                grad = self.wSpikyG(r) / self.restDensity / mem.mass[x1]
                sum_Ci             += grad
                sum_grad_pk_Ci_sqr += grad.norm_sqr()
                rho_i              += self.wPoly6(r.norm_sqr())
            C_i = (mem.mass[x1] * rho_i / self.restDensity) - 1
            sum_grad_pk_Ci_sqr += sum_Ci.norm_sqr()
            self.lambdas[x1] = -C_i / (sum_grad_pk_Ci_sqr + self.relaxation)



    @ti.kernel
    def calc_delta(self):
        '''
        s_corr = k*(w_ij/wDeltaQ)**n

        delta_i = SUM_j{ (lambda_i + lambda_j + s_corr)*wSpikyG_ij } / rho0
        '''
        mem  = self.mem
        grid = self.grid
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            # accumulators
            delta = vec2()
            for j in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, j]
                if mem.phase[x2] == SMOKE: continue 
                r      = mem.newPos[x1] - mem.newPos[x2]
                s_corr = -self.s_corr_k * (self.wPoly6(r.norm_sqr()) * self.s_corr_const) ** self.s_corr_n
                delta += (self.lambdas[x1] + self.lambdas[x2] + s_corr) * self.wSpikyG(r)
            self.deltaX[x1] = delta

    @ti.kernel
    def project(self):
        mem = self.mem
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            mem.newPos[x1] += self.deltaX[x1] / self.restDensity / mem.mass[x1]

    
    
    def add(self, particle:Particle):
        idx = self._size[None]
        self.ptr[idx] = particle.id
        self._size[None] += 1

    def clear(self):
        self._size[None] = 0

    @ti.func
    def size(self):
        return self._size[None]

    @ti.func
    def wPoly6(self, r_sqr):
        ''' poly6 kernel '''
        ret_val = 0.
        if r_sqr < self.kernel2:
            ret_val = self.poly6_const * (self.kernel2 - r_sqr) ** 3
        return ret_val


    @ti.func
    def wSpikyG(self, r):
        ''' spiky gradient kernel '''
        ret_val = vec2()
        r_norm  = r.norm()
        if 0 < r_norm < self.kernel_size:
            ret_val = r / r_norm * self.spikyG_const * (self.kernel_size - r_norm) ** 2
        return ret_val
        