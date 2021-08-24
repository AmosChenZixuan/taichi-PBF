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

class FluidScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))

    def build(self):
        sim = self.sim
        mem = sim.mem
        solver = sim.solvers[STANDARD][FLUID]
        mem.newMesh()
        for i in range(30):
            for j in range(30):
                x = 10 + j * 0.4 * sim.grid_size
                y = 5 + i * 0.4 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=1., phase=FLUID)
                mem.add(p)
                solver.add(p)

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
        sim = self.sim
        mem = sim.mem
        # 1
        solver = sim.solvers[SHAPE][0]
        mem.newMesh()
        for i in range(10):
            for j in range(10):
                x = 300 + j * 0.25 * sim.grid_size
                y = 500 + i * 0.25 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=.5, phase=SOLID)
                mem.add(p)
                solver.add(p)
        solver.init()
        # 2
        solver = sim.solvers[SHAPE][1]
        mem.newMesh()
        for i in range(10):
            for j in range(10):
                x = 500 + j * 0.25 * sim.grid_size
                y = 400 + i * 0.25 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=1.5, phase=SOLID)
                mem.add(p)
                solver.add(p)
        solver.init()

class FluidRigidScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[CONTACT].append(RegularContactSolver(sim.mem, sim.collision_eps))
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1/5)) 
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1))

    def build(self):
        sim = self.sim
        mem = sim.mem
        # water
        solver = sim.solvers[STANDARD][FLUID]
        mem.newMesh()
        for i in range(60):
            for j in range(60):
                x = 10 + j * 0.4 * sim.grid_size
                y = 5 + i * 0.4 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=1., phase=FLUID)
                mem.add(p)
                solver.add(p)
        # 1
        solver = sim.solvers[SHAPE][0]
        mem.newMesh()
        for i in range(10):
            for j in range(10):
                x = 300 + j * 0.25 * sim.grid_size
                y = 100 + i * 0.25 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=.5, phase=SOLID)
                mem.add(p)
                solver.add(p); sim.solvers[STANDARD][FLUID].add(p)
        solver.init()
        # 2
        solver = sim.solvers[SHAPE][1]
        mem.newMesh()
        for i in range(10):
            for j in range(10):
                x = 500 + j * 0.25 * sim.grid_size
                y = 100 + i * 0.25 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=.9, phase=SOLID)
                mem.add(p)
                solver.add(p); sim.solvers[STANDARD][FLUID].add(p)
        solver.init()


GALLARY = [FluidScene, GasScene, BodyScene, FluidRigidScene]
