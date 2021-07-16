from core.memory import DevMemory
import taichi as ti
from include import *

@ti.data_oriented
class SpatialHasher:
    def __init__(self, memory: DevMemory,
            grid_size,
            grid_shape,
            grid_max     = 64,
            neighbor_max = 64,
            capacity = MEM_CAPACITY):
        # reference
        self.mem = memory
        # const
        self.grid_size     = grid_size
        self.grid2         = grid_size**2
        self.grid_shape    = grid_shape
        self.grid_max      = grid_max
        self.neighbors_max = neighbor_max
        # fields
        self.grid       = new_field((*grid_shape, grid_max), 1, INDEX_TYPE)  # grid representation of positions
        self.n_in_grid  = new_field(grid_shape, 1, COUNTER_TYPE)             # number of particle in grid cell
        self.neighbors  = new_field((capacity, neighbor_max), 1, INDEX_TYPE) # neighbor search table
        self.n_neighbors= new_field(capacity, 1, COUNTER_TYPE)               # number of neighbors for each particle
        
    def step(self):
        self.clear()
        self.gridHashing()
        self.neighborSearch() 

    @ti.kernel
    def clear(self):
        # (1) clear grid; erase all counters
        for i, j in self.n_in_grid:
            self.n_in_grid[i,j] = 0

    @ti.kernel
    def gridHashing(self):
        # (2) reconstruct grid representation for current state
        # TODO: hash and reorder
        mem = self.mem
        for i in range(mem.size()):
            # skip dead
            if not mem.lifetime[i]: continue
            grid_idx = int(mem.newPos[i] / self.grid_shape)
            n = ti.atomic_add(self.n_in_grid[grid_idx], 1)
            if (n < self.grid_max):
                self.grid[grid_idx[0], grid_idx[1], n] = i

    @ti.kernel
    def neighborSearch(self):
        # (3) update neighbour table; look up 9th grids around each particle
        #  ----------
        #  | 0| 1| 2|
        #  ----------
        #  | 3| p| 5|
        #  ----------
        #  | 6| 7| 8|
        #  ----------
        mem = self.mem
        for x1 in range(mem.size()):
            # skip dead
            if not mem.lifetime[i]: continue
            neighbor_idx = 0
            gridi, gridj = int(mem.newPos[x1] / self.grid_size)
            for dy in ti.static(range(-1,2)):
                y = gridj + dy
                if 0 <= y < self.grid_shape[1]:
                    for dx in ti.static(range(-1,2)):
                        x = gridi + dx
                        if 0 <= x < self.grid_shape[0]: 
                            for i in range(self.n_in_grid[x,y]):
                                x2 = self.grid[x,y,i]
                                if (neighbor_idx >= self.neighbors_max): break
                                if (x1 != x2 and (mem.newPos[x2] - mem.newPos[x1]).norm_sqr() < self.grid2):
                                    self.neighbors[x1, neighbor_idx] = x2
                                    neighbor_idx += 1
            self.n_neighbors[x1] = neighbor_idx