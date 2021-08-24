import taichi as ti
import numpy as np
from core.memory import DevMemory
from core.spatialHasing import SpatialHasher
from include import *

@ti.data_oriented
class fluidSolver:
    def __init__(self, memory: DevMemory, grid:SpatialHasher, h):
        self.mem   = memory
        self.grid  = grid
        self.ptr   = new_field(memory.capacity, 1, INDEX_TYPE) # global index
        self._size = new_field((), 1, COUNTER_TYPE)            # number of particles bounbed by this constraint 
        # const
        self.kernel_size = h                   # h value for kernels
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
        # vorticirty & vicosity
        self.vort_eps    = 200
        self.visc_c      = 0.01
        # Surface Tension
        self.gamma       = 2e6
        self.cohes_const = 32 / np.pi / self.kernel_size**9
        self.curv_scale  = 0.0001
        # Solid Contact
        self.solid_pressure = .7
        
    def solve(self):
        self.update_cache()
        self.calc_lambda()
        self.calc_delta()
        self.project()

    @ti.kernel
    def update_cache(self):
        mem  = self.mem
        grid = self.grid
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            for j in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, j]
                r  = mem.newPos[x1] - mem.newPos[x2]
                mem.polyBuf[x1,j] = self.wPoly6(r.norm_sqr())
                mem.spkyBuf[x1,j] = self.wSpikyG(r)

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
            rho0               = self.restDensity * mem.mass[x1]
            rho_i              = self.wPoly6(0.)
            sum_Ci             = vec2()
            sum_grad_pk_Ci_sqr = 0.
            for j in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, j]
                grad = mem.spkyBuf[x1,j] #/ (self.restDensity * mem.mass[x2])
                sum_Ci             += grad
                sum_grad_pk_Ci_sqr += grad.norm_sqr()
                if mem.phase[x2] == SOLID:
                    rho_i              += self.solid_pressure * mem.polyBuf[x1,j] * mem.mass[x2]
                else:
                    rho_i              += mem.polyBuf[x1,j] * mem.mass[x2]
            C_i = (rho_i / rho0) - 1
            sum_grad_pk_Ci_sqr += sum_Ci.norm_sqr()
            mem.lambdas[x1] = -C_i / (sum_grad_pk_Ci_sqr + self.relaxation)
            # cache for later computation
            mem.density[x1] = rho_i



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
                r      = mem.newPos[x1] - mem.newPos[x2]
                s_corr = -self.s_corr_k * (mem.polyBuf[x1,j] * self.s_corr_const) ** self.s_corr_n
                delta += (mem.lambdas[x1] + mem.lambdas[x2] + s_corr) * mem.spkyBuf[x1,j]
            mem.deltaX[x1] = delta

    @ti.kernel
    def project(self):
        mem = self.mem
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            mem.newPos[x1] += mem.deltaX[x1] / self.restDensity / mem.mass[x1]

    ##
    ## External Forces
    ##

    def external_forces(self):
        #self.update_cache()
        self.vorticity_confinement() # artificial curl force
        self.xsphViscosity()         # artificial damping
        self.calcNormals()           # prepare for calculating curvature
        self.applySurfaceTension()   

    @ti.kernel
    def vorticity_confinement(self):
        '''
            fvort = eps * (N x omega)
        '''
        mem  = self.mem
        grid = self.grid
        for xi in range(self.size()):
            x1 = self.ptr[xi]
            if not mem.lifetime[x1]: continue
            if mem.phase[x1] > GAS : continue
            # angular velocity
            omega = vec2()
            for i in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, i]
                vel_diff = mem.velocity[x2] - mem.velocity[x1]
                grad     = mem.spkyBuf[x1,i]
                omega   += vel_diff.cross(grad)
            # direction of corrective force
            eta = vec2()
            for i in range(grid.n_neighbors[x1]):
                x2   = grid.neighbors[x1, i]
                grad = mem.spkyBuf[x1,i]
                eta += grad * omega.norm()
            # update if there is an eta direction
            if eta.norm() > 0:
                n = eta.normalized()
                mem.velocity[x1] += n.cross(omega) * self.vort_eps

    @ti.kernel
    def xsphViscosity(self):
        '''
            v_new = v + c * SUM_J{ Vij * poly(Pij) }
        '''
        mem  = self.mem
        grid = self.grid
        for xi in range(self.size()):
            x1 = self.ptr[xi]
            if not mem.lifetime[x1]: continue
            if mem.phase[x1] > GAS : continue
            visc = vec2()
            for i in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, i]
                vel_diff = mem.velocity[x2] - mem.velocity[x1]
                vel_diff *= mem.polyBuf[x1,i]
                visc     += vel_diff
            mem.velocity[x1] += visc * self.visc_c

    @ti.kernel
    def calcNormals(self):
        mem = self.mem
        grid = self.grid
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            if mem.phase[x1] > GAS : continue
            norm = vec2()
            for j in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, j]
                if mem.phase[x1] != mem.phase[x2]:continue  # need to be same type of particle
                if mem.density[x2] > 0:
                    norm += mem.mass[x2] / mem.density[x2] * mem.spkyBuf[x1,j]
            mem.normals[x1] = norm * self.curv_scale


    @ti.kernel
    def applySurfaceTension(self):
        '''
            F_cohesion = -gamma * massi * massj * C(|Xij|) * (Xij/|Xij|)
            F_surfaceTension = 2 * rho_0 / (rh_oi + rho_j) * (F_cohesion + F_curvature)
        '''
        mem = self.mem
        grid = self.grid
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            if mem.phase[x1] > GAS : continue
            force = vec2()
            for j in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, j]
                if mem.phase[x1] != mem.phase[x2]:continue  # need to be same type of particle
                r  = mem.newPos[x1] - mem.newPos[x2]
                rn = r.norm()
                d  = mem.density[x1] + mem.density[x2]
                # Cohesion and Curvature
                if rn > 0 and d > 0:
                    d       = 2 * self.restDensity * mem.mass[x1] / d
                    f_cohes = -self.gamma * mem.mass[x1] * mem.mass[x2] * self.wCohesion(rn) * r.normalized()
                    curv    = -self.gamma * mem.mass[x1] * (mem.normals[x1] - mem.normals[x2])
                    force  += d * (f_cohes + curv) 
            mem.force[x1] += force


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
        
    @ti.func
    def wCohesion(self, r_norm):
        ret_val = 0.
        if 2*r_norm > self.kernel_size and r_norm <= self.kernel_size:
            ret_val = self.cohes_const * (self.kernel_size - r_norm)**3 * r_norm**3
        elif r_norm > 0 and 2*r_norm <= self.kernel_size:
            ret_val = self.cohes_const * (2 * (self.kernel_size - r_norm)**3 * r_norm**3 - (self.kernel_size**6 / 64))
        return ret_val