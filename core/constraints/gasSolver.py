import taichi as ti
import numpy as np
from core.memory import DevMemory
from core.spatialHasing import SpatialHasher
from core.constraints.fluidSolver import fluidSolver
from include import *

@ti.data_oriented
class gasSolver(fluidSolver):
    def __init__(self, memory: DevMemory, grid:SpatialHasher, h):
        super().__init__(memory, grid, h)
        # vorticirty & vicosity
        self.vort_eps = 400
        self.visc_c   = 50
        # surface tension
        self.gamma    = 2e7
        # drag
        self.drag_k   = 0.001


    ##
    ## External Forces
    ##

    def external_forces(self):
        #self.vorticity_confinement() # artificial curl force
        self.xsphViscosity()         # artificial damping
        self.applySurfaceTension()   
        self.drag_force()
        #self.baroclinic_turbulence()


    @ti.kernel
    def drag_force(self):
        '''
            f = -k(v_i - v_env) * (1 - rho_i/rho_0)
            v_env is set to 0 to model still air
        '''
        mem  = self.mem
        for i in range(self.size()):
            x1 = self.ptr[i]
            if not mem.lifetime[x1]: continue
            fdrag = -self.drag_k * (mem.velocity[x1] - 0) * (1 - mem.density[x1] / self.restDensity / mem.mass[x1] )
            mem.force[x1] += fdrag
            

        