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
        self._cycle = 0
        self.paused  = False
        # sim
        self.substeps           = 2
        self.solver_iters       = 5
        self.dt                 = 1 / 60 / self.substeps
        self.gravity            = vec2(y=-980.)
        self.alpha              = 0.005         # gravity refactor for gas
        self.collision_eps      = 4
        self.bbox               = 0,100,0,100
        # Memory
        self.mem = DevMemory()
        # renderer
        self.renderer = None
        # grid
        self.grid_size = 25
        self.grid = SpatialHasher()
        # constraints
        self.solvers = newConstraintGroups()
        # scene
        self.cur_scene = None

    def tick(self, amount=0):
        self._ticks += amount
        return self._ticks

    def register_externals(self, renderer: Renderer):
        self.renderer = renderer
        renderer.registrating(self.mem)
        self.bbox     = 0, renderer.window[0], 0, renderer.window[1]-10
        # intialize spatial hasher
        w,h = renderer.window
        grid_shape = (w // self.grid_size + 1, 
                        h // self.grid_size + 1)
        self.grid.initialize(self.mem, self.grid_size, grid_shape)

    def set_scene(self, scene):
        self.cur_scene = scene
        self._cycle = 0
        self.solvers = newConstraintGroups()

    def reset(self):
        self.mem.clear()
        if self.cur_scene:
            self.cur_scene.load()
            self._ticks = 0
            self._cycle += 1

    def emit_smoke(self):
        gas_row, gas_col = 3, 3
        smk_row, smk_col = 6, 3
        life = 1500
        # gas
        try:
            solver = self.solvers[STANDARD][GAS]
        except:
            return
        self.mem.newMesh()
        for i in range(gas_row):
            for j in range(gas_col):
                x = 290 + j * 10
                y = 10 + i * 10
                v = vec2(0, 200 - 180 * (abs(1-j)))
                p = Particle(self.mem.getNextId(), [x,y], vel = v, mass=1., lifetime=life, phase=GAS)
                self.mem.add(p)
                solver.add(p)
        # smoke
        self.mem.newMesh()
        for i in range(smk_row):
            for j in range(smk_col):
                x = 298 + j * 2
                y = 12.5 + i * 3
                p = Particle(self.mem.getNextId(), [x,y], lifetime=life, phase=SMOKE)
                self.mem.add(p)

    @ti.kernel
    def apply_force(self, mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
        mem = self.mem
        for i in range(mem.size()):
            # skip dead or visual particles
            if not mem.lifetime[i]: continue
            if mem.phase[i] == SMOKE: continue
            if mem.mass[i] == 0.: continue
            # v1 = v0 + f*dt 
            g = self.gravity
            if mem.phase[i] == GAS:
                g *= self.alpha
            mem.velocity[i] += self.dt * g + mem.force[i]
            # reset acceleration
            mem.force[i] = 0,0
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


    @ti.kernel
    def update_contacts(self):
        mem  = self.mem
        grid = self.grid
        solver = self.solvers[CONTACT][0]
        solver.clear()
        for x1 in range(mem.size()):
            # skip dead or visual particles
            if not mem.lifetime[x1]: continue
            if mem.phase[x1] == SMOKE: continue
            for i in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, i]
                # skip if same mesh
                if mem.mesh[x1] == mem.mesh[x2]:
                    continue
                r  = mem.newPos[x1] - mem.newPos[x2]
                if (mem.phase[x1] == SOLID or mem.phase[x2] == SOLID) and r.norm() < self.collision_eps:
                    solver.add(x1, x2)

    def project(self, substep, iter):
        for i, group in enumerate(self.solvers):
            # shapematching only project once at the last iteration
            if i == SHAPE and (iter < self.solver_iters-1): # or substep < self.substeps-1):
                continue
            for solver in group:
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
                mem.newPos[i][0] = l #+ ti.random()
            if mem.newPos[i][0] > r:
                mem.newPos[i][0] = r  #- ti.random()
            if mem.newPos[i][1] < b:
                mem.newPos[i][1] = b #+ ti.random()
            if mem.newPos[i][1] > t:
                mem.newPos[i][1] = t #- ti.random()

    @ti.kernel
    def update(self):
        mem  = self.mem
        for i in range(mem.size()):
            if not mem.lifetime[i]: continue
            mem.lifetime[i] -= 1
            if mem.phase[i] == SMOKE: continue
            if mem.mass[i] == 0.: continue
            mem.velocity[i] = (mem.newPos[i] - mem.curPos[i]) / self.dt * 0.99
            mem.curPos[i]   = mem.newPos[i]

    @ti.kernel
    def advect_smoke(self):
        mem  = self.mem
        grid = self.grid
        # Advect Smoke
        for x1 in range(mem.size()):
            if not mem.lifetime[x1]: continue
            if mem.phase[x1] != SMOKE: continue
            vsum = vec2()
            wsum = 0.
            for i in range(grid.n_neighbors[x1]):
                x2 = grid.neighbors[x1, i]
                r  = mem.curPos[x1] - mem.curPos[x2]
                w  = self.solvers[STANDARD][GAS].wPoly6(r.norm_sqr())
                vsum += w * mem.velocity[x2]
                wsum += w
            if wsum > 0:
                mem.curPos[x1] += vsum / wsum * self.dt
                mem.newPos[x1] = mem.curPos[x1]
            else:
                # remove smoke particles with no fluid neighbor
                mem.lifetime[x1] = 0

    def external_forces(self):
        try:
            fs = self.solvers[STANDARD][FLUID]
            gs = self.solvers[STANDARD][GAS]
        except:
            return
        fs.external_forces()
        gs.external_forces()

    def step(self):
        if self.paused:
            return
        for i in range(self.substeps):
            # time integration - semi-implicit
            x,y = self.renderer.mouse_pos
            self.apply_force(x,y, self.renderer.attract)
            self.box_confinement()
            # update grid info
            self.grid.step()
            # add collision
            if len(self.solvers[CONTACT]) > 0:
                self.update_contacts()
            # non-linear Jacobi Iteration
            for j in range(self.solver_iters): 
                self.project(i, j)
            # update v and pos
            self.update()
            # advect smoke
            self.advect_smoke()
            # additional forces
            self.external_forces()

    @ti.kernel
    def pick(self, idx:ti.i32, mouse_x: ti.f32, mouse_y: ti.f32):
        x1  = vec2(mouse_x*600, mouse_y*1000)
        self.mem.curPos[idx] = x1
