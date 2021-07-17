import taichi as ti
from include import *
from core import DevMemory, SpatialHasher
from core.constraints import *
from src.renderer import Renderer

@ti.data_oriented
class Simulation:
    def __init__(self):
        # control
        self._ticks = 0
        self.paused  = False
        # sim
        self.substeps           = 2
        self.solver_iters       = 5
        self.dt                 = 1 / 60 / self.substeps
        self.gravity            = vec2(y=-980.)
        self.alpha              = -0.2          # gravity refactor for gas
        self.collision_eps      = 5
        self.bbox               = 0,600,0,590
        # Memory
        self.mem = DevMemory()
        # renderer
        self.renderer = None
        # grid
        self.grid_size = 25
        self.grid = SpatialHasher()
        # constraints
        self.solvers = [fluidSolver(self.mem, self.grid)]

    def tick(self, amount=0):
        self._ticks += amount
        return self._ticks

    def register_externals(self, renderer: Renderer):
        self.renderer = renderer
        renderer.registrating(self.mem)
        # intialize spatial hasher
        w,h = renderer.window
        grid_shape = (w // self.grid_size + 1, 
                        h // self.grid_size + 1)
        self.grid.initialize(self.mem, self.grid_size, grid_shape, 64, 64)

    def reset(self):
        # reinitialize
        mem = self.mem
        mem.clear()
        for s in self.solvers:
            s.clear()
        # add water
        solver = self.solvers[FLUID]
        for i in range(130):
            for j in range(30):
                x = 220 + j * 5
                y = 15 + i * 5
                p = Particle(mem.getNextId(), [x,y], mass=1., phase=FLUID)
                mem.add(p)
                # TODO add fluid contraints
                solver.add(p)


    def emit_smoke(self):
        gas_row, gas_col = 2, 3
        smk_row, smk_col = 1, 1
        # gas
        for i in range(gas_row):
            for j in range(gas_col):
                x = 290 + j * 15
                y = 20 + i * 15
                p = Particle(self.mem.getNextId(), [x,y], mass=1.5, phase=GAS)
                self.mem.add(p)
                # TODO add gas contraints
        # smoke
        for i in range(smk_row):
            for j in range(smk_col):
                x = 290 + j * 2
                y = 20 + i * 2
                p = Particle(self.mem.getNextId(), [x,y], lifetime=1000, phase=SMOKE)
                self.mem.add(p)

    @ti.kernel
    def apply_force(self, mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
        mem = self.mem
        for i in range(mem.size()):
            # skip dead or visual particles
            if not mem.lifetime[i]: continue
            if mem.phase[i] == SMOKE: continue
            # v1 = v0 + f*dt 
            g = self.gravity
            if mem.phase[i] == GAS:
                g *= self.alpha
            mem.velocity[i] += self.dt * (g + mem.force[i])
            # mouse interaction - F = GMm/|r|^2 * (r/|r|)
            if attract:
                w,h = self.renderer.window
                x, y =  mouse_x * w, mouse_y * h
                r = vec2(x, y) - mem.curPos[i]
                r_norm = r.norm()
                if r_norm > 15:
                    mem.velocity[i] += attract * self.dt * 5e6 * r / r_norm ** 3 
            # estimate
            # x1 = x0 + v1*dt
            mem.newPos[i] = mem.curPos[i] + self.dt * mem.velocity[i]

    def project(self):
        for solver in self.solvers:
            solver.solve()

    @ti.kernel
    def box_confinement(self):
        mem = self.mem
        l,r,b,t = self.bbox
        l += self.collision_eps; b += self.collision_eps
        r -= self.collision_eps; t -= self.collision_eps
        for i in range(mem.size()):
            if not mem.lifetime[i]: continue
            if mem.newPos[i][0] < l:
                mem.newPos[i][0] = l + ti.random()
            if mem.newPos[i][0] > r:
                mem.newPos[i][0] = r  + ti.random()
            if mem.newPos[i][1] < b:
                mem.newPos[i][1] = b + ti.random()
            if mem.newPos[i][1] > t:
                mem.newPos[i][1] = t + ti.random()

    @ti.kernel
    def update(self):
        mem = self.mem
        for i in range(mem.size()):
            if not mem.lifetime[i]: continue
            if mem.phase[i] == SMOKE: continue
            mem.lifetime[i] -= 1
            mem.velocity[i] = (mem.newPos[i] - mem.curPos[i]) / self.dt * 0.99
            mem.curPos[i]   = mem.newPos[i]
        # # Advect smoke
        # ph = Phase.smoke.value
        # for xi in range(self.mem._size[ph]):
        #     x1 = mem.index[ph, xi]
        #     if not mem.lifetime[x1]: continue
        #     Vsum = vec2()
        #     Wsum = 0.
        #     for i in range(mem.n_neighbors[x1]):
        #         x2 = mem.neighbors[x1, i]
        #         if mem.phase[x2] == Phase.smoke.value: continue
        #         r = mem.curPos[x1] - mem.curPos[x2]
        #         w = self.wPoly6(r.norm_sqr())
        #         Vsum += w * mem.velocity[x2]
        #         Wsum += w
        #     if Wsum > 0.:
        #         mem.velocity[x1] = Vsum / Wsum



    def step(self):
        if self.paused:
            return
        for _ in range(self.substeps):
            # time integration - semi-implicit
            x,y = self.renderer.mouse_pos
            self.apply_force(x,y, self.renderer.attract)
            self.box_confinement()
            # update grid info
            self.grid.step()
            # non-linear Jacobi Iteration
            for _ in range(self.solver_iters): 
                self.project()
            # update v and pos
            self.update()