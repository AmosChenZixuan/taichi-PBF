from include import *
from core.constraints import *

class Scene:
    def __init__(self, sim):
        self.sim = sim

    def load(self):
        sim = self.sim
        if sim._cycle:    # initialized before, reset solvers
            for group in sim.solvers:
                for solver in group:
                    solver.clear()
        else:
            self.initialize()
        self.build()
    
    def initialize(self): raise NotImplementedError()
    def build(self): raise NotImplementedError()

    def add_water(self, xy, w, h, wspacing, hspacing, mass=1.):
        sim = self.sim
        mem = sim.mem
        solver = sim.solvers[STANDARD][FLUID]
        mem.newMesh()
        x0, y0 = xy
        for i in range(w):
            for j in range(h):
                x = x0 + j * wspacing * sim.grid_size
                y = y0 + i * hspacing * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=mass, phase=FLUID)
                mem.add(p)
                solver.add(p)

    def add_block(self, idx, xy, w, h, mass=1.):
        sim = self.sim
        mem = sim.mem
        solver = sim.solvers[SHAPE][idx]
        mem.newMesh()
        x0, y0 = xy
        for i in range(w):
            for j in range(h):
                x = x0 + j * 0.25 * sim.grid_size
                y = y0 + i * 0.25 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=mass, phase=SOLID)
                mem.add(p)
                solver.add(p); sim.solvers[STANDARD][FLUID].add(p)
        solver.init()

class FluidScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))

    def build(self):
        self.add_water((10, 5), 30, 30, 0.4, 0.8, mass=1.)

class GasScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))
    
    def build(self):pass


class BodyScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[CONTACT].append(RegularContactSolver(sim.mem, sim.collision_eps))
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1/5)) 
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1))
    
    def build(self):
        self.add_block(0, (200,500), 10, 10, mass=.5)
        self.add_block(1, (400,500), 10, 10, mass=.9)

class FluidRigidScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[CONTACT].append(RegularContactSolver(sim.mem, sim.collision_eps))
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1/5)) 
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1))

    def build(self):
        self.add_water((10,5), 60, 60, 0.4, 0.4, mass=1.)
        self.add_block(0, (200,100), 10, 10, mass=.5)
        self.add_block(1, (400,100), 10, 10, mass=.9)


GALLARY = [FluidScene, GasScene, BodyScene, FluidRigidScene]
