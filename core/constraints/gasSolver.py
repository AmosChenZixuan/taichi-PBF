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

        