import taichi as ti
from include import *
from core import DevMemory, SpatialHasher
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
        self.grid = None

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
        self.grid = SpatialHasher(self.mem, self.grid_size, grid_shape, 64, 64)

    def reset(self):
        self.mem.clear()
        for i in range(30):
            for j in range(30):
                x = 220 + j * 5
                y = 15 + i * 5
                p = Particle(self.mem.getNextId(), [x,y], mass=1., phase=FLUID)
                self.mem.add(p)
                # TODO add fluid contraints

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
    def apply_force(self):
        mem = self.mem
        rend = self.renderer
        for i in range(mem.size()):
            # skip dead or visual particles
            if not mem.lifetime[i]: continue
            if mem.phase[i] == SMOKE: continue
            # v1 = v0 + f*dt 
            g = self.gravity
            if mem.phase[i] == GAS or True:
                g *= self.alpha
            mem.velocity[i] += self.dt * (g + mem.force[i])
            # mouse interaction - F = GMm/|r|^2 * (r/|r|)
            if rend.attract:
                x, y = rend.mouse_pos * rend.window
                r = vec2(x, y) - mem.curPos[i]
                r_norm = r.norm()
                if r_norm > 15:
                    mem.velocity[i] += rend.attract * self.dt * 5e6 * r / r_norm ** 3 
            # estimate
            # x1 = x0 + v1*dt
            mem.newPos[i] = mem.curPos[i] + self.dt * mem.velocity[i]
 
    @ti.kernel
    def find_neighbours(self):
        return
        mem = self.mem
        # (1) clear grid; erase all counters
        for i, j in mem.n_in_grid:
            mem.n_in_grid[i,j] = 0
        # (2) reconstruct grid representation for current state
        for i in range(mem.size()):
            if not mem.lifetime[i]:
                continue
            gridi, gridj = int(mem.newPos[i] / self.grid_size)
            n = ti.atomic_add(mem.n_in_grid[gridi, gridj], 1)
            if n < self.grid_max_capacity:
                mem.grid[gridi, gridj, n] = i
                #mem.n_in_grid[gridi, gridj] += 1
        # (3) update neighbour table; look up 9th grids around each particle
        #  ----------
        #  | 0| 1| 2|
        #  ----------
        #  | 3| p| 5|
        #  ----------
        #  | 6| 7| 8|
        #  ----------
        for x1 in range(mem.size()):
            if not mem.lifetime[x1]:
                continue
            n_neighbor = 0
            gridi, gridj = int(mem.newPos[x1] / self.grid_size)
            for dy in ti.static(range(-1,2)):
                y = gridj + dy
                if 0 <= y < self.grid_shape[1]:
                    for dx in ti.static(range(-1,2)):
                        x = gridi + dx
                        if 0 <= x < self.grid_shape[0]: 
                            for i in range(mem.n_in_grid[x,y]):
                                x2 = mem.grid[x,y,i]
                                if (n_neighbor >= self.max_neighbors): break  
                                if (x1 != x2 and (mem.newPos[x2] - mem.newPos[x1]).norm_sqr() < self.kernel_sqr):
                                    mem.neighbors[x1, n_neighbor] = x2
                                    n_neighbor += 1
            mem.n_neighbors[x1] = n_neighbor

    @ti.kernel
    def project(self):
        pass
        # for each solver: solve!

    @ti.kernel
    def box_confinement(self):
        mem = self.mem
        l,r,b,t = self.bbox
        l += self.collision_eps; b += self.collision_eps
        r -= self.collision_eps; t -= self.collision_eps
        for i in range(mem.size()):
            if not mem.lifetime[i]: continue
            if mem.curPos[i][0] < l:
                mem.curPos[i][0] = l 
            if mem.curPos[i][0] > r:
                mem.curPos[i][0] = r  
            if mem.curPos[i][1] < b:
                mem.curPos[i][1] = b 
            if mem.curPos[i][1] > t:
                mem.curPos[i][1] = t

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
            self.apply_force()
            # update grid info
            self.find_neighbours()
            # non-linear Jacobi Iteration
            for _ in range(self.solver_iters): 
                self.project()
            # update v and pos
            self.update()
            self.box_confinement()