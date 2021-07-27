import taichi as ti
import numpy as np
from core.memory import DevMemory
from core.spatialHasing import SpatialHasher
from core.constraints.fluidSolver import fluidSolver
from include import *

@ti.data_oriented
class gasSolver(fluidSolver):
    pass

    # @ti.kernel
    # def calc_delta(self):
    #     '''
    #     s_corr = k*(w_ij/wDeltaQ)**n

    #     delta_i = SUM_j{ (lambda_i + lambda_j + s_corr)*wSpikyG_ij } / rho0
    #     '''
    #     mem  = self.mem
    #     grid = self.grid
    #     for i in range(self.size()):
    #         x1 = self.ptr[i]
    #         if not mem.lifetime[x1]: continue
    #         # accumulators
    #         delta      = vec2()
    #         curl_force = vec2()
    #         for j in range(grid.n_neighbors[x1]):
    #             x2 = grid.neighbors[x1, j]
    #             if mem.phase[x2] == SMOKE: continue 
    #             r      = mem.newPos[x1] - mem.newPos[x2]
    #             s_corr = -self.s_corr_k * (self.wPoly6(r.norm_sqr()) * self.s_corr_const) ** self.s_corr_n
    #             delta += (mem.lambdas[x1] + mem.lambdas[x2] + s_corr) * self.wSpikyG(r)
    #             # # compute gas curl force
    #             # grad        = self.wSpikyG(r)
    #             # w           = grad * mem.velocity[x2]
    #             # cross       = vec3(z=w.norm()).cross(vec3(r[0], r[1]))
    #             # curl_force += vec2(cross[0], cross[1]) * self.wPoly6(r.norm_sqr())
    #         mem.deltaX[x1] = delta
    #         mem.force[x1] += curl_force

        