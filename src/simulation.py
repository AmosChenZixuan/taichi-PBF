import taichi as ti
from src.vector import *
from include import *

@ti.data_oriented
class Simulation:
    def __init__(self):
        # control
        self.paused = False
        self.display_fluid = True
        self.attract = 0
        self.mouse_pos = 0,0
        self.tick = 0
        # sim
        self.substeps           = 2
        self.solver_iters       = 5
        self.dt = 1 / 60 / self.substeps
        self.gravity = vec2(y=-980.)
        self.collision_eps      = 5