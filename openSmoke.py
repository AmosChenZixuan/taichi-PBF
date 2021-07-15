import taichi as ti
import numpy as np
from enum import Enum, unique
from time import perf_counter as clock
ti.init(arch=ti.gpu)

'''
0. Fluid particle: Basic unit in PBF
1. Gas Particle: Basically Fluid, participate in computation
2. Smoke Particle: Injected visually, advect passively. No computation
'''
@unique
class Phase(Enum):
    fluid = 0
    gas = 1
    smoke = 2

@ti.data_oriented
class Mem:
    PHASE_TYPE = ti.i8
    INDEX_TYPE = ti.i32
    COUNTER_TYPE = ti.i32
    def __init__(self, capacity, grid_shape, grid_max, neighbor_max):
        self.capacity   = capacity  # max number of particles
        self.total_size  = 0         # cur number of particles
        self._size       = self.new_field(len(Phase), 1, Mem.COUNTER_TYPE) # number of particles of each phase
        # PBDynamics
        self.curPos     = self.new_field(capacity) # X, true position
        self.newPos     = self.new_field(capacity) # P, estimated position
        self.velocity   = self.new_field(capacity) # V, velocity
        self.force      = self.new_field(capacity) # F/Acceleration, other than gravity
        self.mass       = self.new_field(capacity, 1) 
        self.phase      = self.new_field(capacity, 1, Mem.PHASE_TYPE)
        self.lifetime   = self.new_field(capacity, 1, Mem.COUNTER_TYPE)          # [-1]not applicable; [0]dead; [>0]alive
        self.index      = self.new_field((len(Phase), capacity), 1, Mem.INDEX_TYPE) # using size[i,j] to fetch jth idx of phase i
        # PBFluid
        self.lambdas    = self.new_field(capacity, 1)  # constraint
        self.deltaX     = self.new_field(capacity)     # position change
        self.grid       = self.new_field((*grid_shape, grid_max), 1, Mem.INDEX_TYPE) # grid representation of positions
        self.n_in_grid  = self.new_field(grid_shape, 1, Mem.COUNTER_TYPE)     # number of particle in grid cell
        self.neighbors  = self.new_field((capacity, neighbor_max), 1, Mem.INDEX_TYPE) # neighbor search table
        self.n_neighbors= self.new_field(capacity, Mem.COUNTER_TYPE)          # number of neighbors for each particle
        # Render
        self.p2Render   = self.new_field(capacity)
        # Done
        self.clear()                # init/reset memory counters


    @staticmethod
    def new_field(shape, dim=2, dtype=ti.f32):
        ''' Allocate Memory on Device '''
        if dim == 2:
            return ti.Vector.field(2, dtype=dtype, shape=shape)
        else:
            return ti.field(dtype, shape=shape)
        
    def clear(self):
        ''' Resetting counters. Old data will be overwritten as the new simulation proceed '''
        self.total_size  = 0         # cur number of particles
        self._size.fill(0)

    def add(self, pos, v, f, m, lf, ph):
        i,j = self.total_size, self._size[ph.value]
        self.curPos[i]  = pos
        self.velocity[i]= v
        self.force[i]   = f
        self.mass[i]    = m
        self.phase[i]   = ph.value
        self.lifetime[i]= lf
        self.index[ph.value, j] = i
        # increment counters
        self.total_size += 1
        self._size[ph.value] += 1

    @ti.func
    def size(self, ph=False):
        ret = self.total_size
        # if ph != False:
        #     ret = self._size[ph]
        return ret



@ti.data_oriented
class Simulation:
    def __init__(self):
        # GUI
        self.window         = 600,600
        self.box            = 0,600,0,590
        self.bg_color       = 0xf0f0f0
        self.palette        = np.array([0x328ac1, 0xff1c8a, 0x959595])
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
        # grid 
        self.grid_max_capacity  = 64
        self.max_neighbors      = self.grid_max_capacity
        self.kernel_size        = 25                        # h vale for kernels
        self.kernel_sqr         = self.kernel_size**2
        self.grid_size          = self.kernel_size
        self.grid_shape         = (self.window[0] // self.grid_size + 1, 
                                    self.window[1] // self.grid_size + 1)
        # fluid
        self.poly6_const        = 315 / 64 / np.pi / self.kernel_size**9
        self.spikyG_const       = -45 / np.pi / self.kernel_size**6
        self.restDensity        = (self.poly6_const * self.kernel_sqr**3) * 0.5 # rho_0 = restDensity / inv_mass
        self.relaxation         = 500       # applied to lambda 
        # gas
        self.alpha              = -0.2      # gravity refactor for gas
        # Tensile Instability  (repulsive term S_corr)
        deltaQ = 0.25 * self.kernel_size    # 0.1h ~ 0.3h
        self.s_corr_k           = 0.1       # s_corr = k*(w_ij/wDeltaQ)**n
        self.s_corr_n           = 4
        self.s_corr_const       = 1 / (self.poly6_const * (self.kernel_sqr - deltaQ**2) ** 3)
        # memory
        self.mem = Mem(2**13, self.grid_shape, self.grid_max_capacity, self.max_neighbors)


    @ti.func
    def wPoly6(self, r_sqr):
        ''' poly6 kernel '''
        ret_val = 0.
        if r_sqr < self.kernel_sqr:
            ret_val = self.poly6_const * (self.kernel_sqr - r_sqr) ** 3
        return ret_val


    @ti.func
    def wSpikyG(self, r):
        ''' spiky gradient kernel '''
        ret_val = vec2()
        r_norm  = r.norm()
        if 0 < r_norm < self.kernel_size:
            ret_val = r / r_norm * self.spikyG_const * (self.kernel_size - r_norm) ** 2
        return ret_val


    def reset(self):
        self.mem.clear()
        for i in range(30):
            for j in range(30):
                x = 220 + j * self.kernel_size *0.2
                y = 15 + i * self.kernel_size * 0.2
                v = 0.,0.
                f = 0.,0.
                mass = 1.
                life = -1
                phase = Phase.fluid
                self.mem.add([x,y], v, f, mass, life, phase)



    def emit_smoke(self):
        gas_row, gas_col = 2, 3
        smk_row, smk_col = 1, 1
        # gas
        for i in range(gas_row):
            for j in range(gas_col):
                x = 290 + j * self.kernel_size * 0.7
                y = 20 + i * self.kernel_size * 0.7
                v = 0.,0.
                f = 0.,0.
                mass = 0.5
                life = -1
                phase = Phase.gas
                self.mem.add([x,y], v, f, mass, life, phase)

        # smoke
        for i in range(smk_row):
            for j in range(smk_col):
                x = 290 + j * self.kernel_size * 0.1
                y = 20 + i * self.kernel_size * 0.1
                v = 0.,0.
                f = 0.,0.
                mass = 0.5
                life = -1
                phase = Phase.smoke
                self.mem.add([x,y], v, f, mass, life, phase)

    @ti.kernel
    def apply_force(self):
        mem = self.mem
        for i in range(mem.size()):
            if not mem.lifetime[i]: continue
            if mem.phase[i] == Phase.smoke.value: continue
            # v1 = v0 + f*dt 
            g = self.gravity
            if mem.phase[i] == Phase.gas.value:
                g *= self.alpha
            mem.velocity[i] += self.dt * (g + mem.force[i])
            # mouse interaction - F = GMm/|r|^2 * (r/|r|)
            if self.attract:
                x, y = self.mouse_pos
                w, h = self.window
                r = ti.Vector([x * w, y * h]) - mem.curPos[i]
                r_norm = r.norm()
                if r_norm > 15:
                    mem.velocity[i] += self.attract * self.dt * 5e6 * r / r_norm ** 3 
            # estimate
            # x1 = x0 + v1*dt
            mem.newPos[i] = mem.curPos[i] + self.dt * mem.velocity[i]

    @ti.kernel
    def find_neighbours(self):
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
    def project(self, ph:ti.i32):
        mem = self.mem
        # calc lambdas
        for xi in range(self.mem._size[ph]):
            x1 = mem.index[ph, xi]
            if not mem.lifetime[x1]: continue
            sum_SpikyG      = vec2()
            sum_SpikyG_sq   = 0.
            rho_i           = self.wPoly6(0)
            for i in range(mem.n_neighbors[x1]):
                x2 = mem.neighbors[x1, i]
                if mem.phase[x2] == Phase.smoke.value: continue 
                r                = mem.newPos[x1] - mem.newPos[x2]
                grad             = self.wSpikyG(r) / self.restDensity / mem.mass[x1]
                sum_SpikyG      += grad
                sum_SpikyG_sq  += grad.norm_sqr()
                rho_i           += self.wPoly6(r.norm_sqr())
            C_i = rho_i / self.restDensity / mem.mass[x1] - 1
            mem.lambdas[x1] = -C_i / (sum_SpikyG_sq + sum_SpikyG.norm_sqr() + self.relaxation)
        # calc delta
        for xi in range(self.mem._size[ph]):
            x1 = mem.index[ph, xi]
            if not mem.lifetime[x1]: continue
            mem.deltaX[x1]  = vec2()
            fvort           = vec2()
            for i in range(mem.n_neighbors[x1]):
                x2 = mem.neighbors[x1, i]
                if mem.phase[x2] == Phase.smoke.value: continue 
                r = mem.newPos[x1] - mem.newPos[x2]
                s_corr = -self.s_corr_k * (self.wPoly6(r.norm_sqr()) * self.s_corr_const) ** self.s_corr_n
                mem.deltaX[x1] += (mem.lambdas[x1] + mem.lambdas[x2] + s_corr) * self.wSpikyG(r)
                # gas curl
                if mem.phase[x1] == Phase.gas.value:
                    grad   = self.wSpikyG(r) / self.restDensity / mem.mass[x1]
                    w      = grad * mem.velocity[x2]
                    cross  = vec3(z=w.norm_sqr()).cross(vec3(x=r[0], y=r[1]))
                    fvort += vec2(cross[0], cross[1]) * self.wSpikyG(r*r)
            mem.force[x1] = fvort
            #pp(fvort, 1)
        # apply delta
        for xi in range(self.mem._size[ph]):
            x1 = mem.index[ph, xi]
            mem.newPos[x1] += mem.deltaX[x1] / self.restDensity / mem.mass[x1]

    @ti.kernel
    def box_confinement(self):
        mem = self.mem
        l,r,b,t = self.box
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
            if mem.phase[i] == Phase.smoke.value: continue
            mem.lifetime[i] -= 1
            mem.velocity[i] = (mem.newPos[i] - mem.curPos[i]) / self.dt * 0.99
            mem.curPos[i]   = mem.newPos[i]
        # Advect smoke
        ph = Phase.smoke.value
        for xi in range(self.mem._size[ph]):
            x1 = mem.index[ph, xi]
            if not mem.lifetime[x1]: continue
            Vsum = vec2()
            Wsum = 0.
            for i in range(mem.n_neighbors[x1]):
                x2 = mem.neighbors[x1, i]
                if mem.phase[x2] == Phase.smoke.value: continue
                r = mem.curPos[x1] - mem.curPos[x2]
                w = self.wPoly6(r.norm_sqr())
                Vsum += w * mem.velocity[x2]
                Wsum += w
            if Wsum > 0.:
                mem.velocity[x1] = Vsum / Wsum
            
    
    def step(self):
        for _ in range(self.substeps):
            pass
            # time integration - semi-implicit
            self.apply_force()
            # update grid info
            self.find_neighbours()
            # non-linear Jacobi Iteration
            for _ in range(self.solver_iters): 
                self.project(Phase.fluid.value)
                self.project(Phase.gas.value)
            # update v and pos
            self.update()
            self.box_confinement()
        # to host for rendering
        self.copy2Host()

    @ti.kernel
    def copy2Host(self):
        ''' map positions to gui coordinates '''
        w,h = self.window
        for i in range(self.mem.size()):
            x = self.mem.curPos[i]
            self.mem.p2Render[i] = x[0]/w, x[1]/h

    def render(self, gui:ti.template()):
        mem = self.mem
        size = mem.total_size
        active = mem.lifetime.to_numpy()[:size] != 0
        ph = mem.phase.to_numpy()[:size]
        if not self.display_fluid:
            active *= ph != Phase.fluid.value
        pos = mem.p2Render.to_numpy()[:size][active]
        
        gui.circles(pos=pos,
                radius=1.5,
                color=self.palette[ph[active]]
        )


## utils
def vec2(x=0., y=0.):
    return ti.Vector([x, y])

def vec3(x=0., y=0., z=0.):
    return ti.Vector([x, y, z])

@ti.func
def pp(s, i=0):
    print(i,'===',s)

def timeit(c, what):
    print(what, clock() - c)
    return c

        


if __name__ == '__main__':
    sim = Simulation()
    sim.reset()

    gui = ti.GUI('Position Based Fluid',
                 res=sim.window, background_color=sim.bg_color)
    
    while gui.running:
        sim.tick += 1
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == 'p':
                sim.paused = not sim.paused
            elif e.key == 'r':
                sim.reset()
            elif e.key == 'e':
                sim.display_fluid = not sim.display_fluid
            elif e.key == gui.SPACE:
                sim.emit_smoke()
        if False:
            if not (sim.tick % 5):
                sim.emit_smoke()

        if gui.is_pressed(ti.GUI.RMB):
            sim.mouse_pos = gui.get_cursor_pos()
            sim.attract = 1
        elif gui.is_pressed(ti.GUI.LMB):
            sim.mouse_pos = gui.get_cursor_pos()
            sim.attract = -1
        else:
            sim.attract = 0
        print(f'===={sim.tick}====')
        timer = clock()
        if not sim.paused:
            sim.step()
        timer = timeit(timer, 'step')
        sim.render(gui)
        timer = timeit(timer, 'render')
        # Display
        gui.show()
        timer = timeit(timer, 'show')
        