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
        solver   = sim.solvers[STANDARD][FLUID]
        startIdx = mem.getNextId()

        mem.newMesh()
        x0, y0 = xy
        for i in range(w):
            for j in range(h):
                x = x0 + j * wspacing * sim.grid_size
                y = y0 + i * hspacing * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=mass, phase=FLUID)
                mem.add(p)
                solver.add(p)
        return startIdx, mem.getNextId()-1

    def add_block(self, idx, xy, w, h, mass=1., buoyance=True):
        sim = self.sim
        mem = sim.mem
        solver = sim.solvers[SHAPE][idx]
        startIdx = mem.getNextId()

        mem.newMesh()
        x0, y0 = xy
        for i in range(w):
            for j in range(h):
                x = x0 + j * 0.25 * sim.grid_size
                y = y0 + i * 0.25 * sim.grid_size
                p = Particle(mem.getNextId(), [x,y], mass=mass, phase=SOLID)
                mem.add(p)
                solver.add(p)
                if buoyance:
                    sim.solvers[STANDARD][FLUID].add(p)
        solver.init()
        return startIdx, mem.getNextId()-1

    def add_rope(self, begin, end, spacing, mass=1., buoyance=True, stiffness=1.):
        sim = self.sim
        mem = sim.mem
        mem.newMesh()
        startIdx = mem.getNextId()

        x0,y0 = begin
        x1,y1 = end
        n = (vec2(*end)-vec2(*begin)).norm()//spacing
        for i in range(int(n)):
            pi = mem.getNextId()
            dx = i * (x1-x0)/n
            dy = i * (y1-y0)/n
            p = Particle(pi, [x0+dx,y0+dy], mass=mass , phase=CLOTH)
            mem.add(p)
            if buoyance:
                    sim.solvers[STANDARD][FLUID].add(p)
            if i > 0:
                sim.solvers[STANDARD].append(StrechSolver(sim.mem, pi-1, pi, spacing, stiffness))
        return startIdx, mem.getNextId()-1

####   Abstract Scene
############################################################
####   Demo Scene

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

class PendulumScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[SHAPE].append(shapeMatchingSolver(sim.mem, 1))

    def build(self):
        sim = self.sim
        mem = sim.mem
        self.add_water((10, 5), 30, 30, 0.4, 0.8, mass=1.)
        start, end = self.add_rope((300,250), (450, 300), 7, mass=1., buoyance=False)
        mem.mass[start] = 0.
        self.add_block(0, (450,300), 5, 5, mass=2.)
        sim.solvers[STANDARD].append(StrechSolver(sim.mem, end, end+3, 5, 1))

class RopeScene(Scene):
    def initialize(self):
        sim = self.sim
        sim.solvers[STANDARD].append(fluidSolver(sim.mem, sim.grid, sim.grid_size))
        sim.solvers[STANDARD].append(  gasSolver(sim.mem, sim.grid, sim.grid_size))

    def build(self):
        sim = self.sim
        self.add_rope((300,950), (301, 350), 7, mass=1., buoyance=True)



GALLARY = [FluidScene, GasScene, PendulumScene, RopeScene, BodyScene, FluidRigidScene]
DEFAULT_SCENE = RopeScene