import taichi as ti
import numpy as np
from include import *
from core import DevMemory

@ti.data_oriented
class Renderer:
    def __init__(self, title='Position Based Fluid'):
        self.mouse_pos = vec2()
        self.attract = 0
        self.display_fluid = True
        # GUI
        self.window        = 600,600
        self.bg_color      = 0xf0f0f0
        self.gui = ti.GUI(title,
                res=self.window, 
                background_color=self.bg_color
                )
        # data reference
        self.mem = None   # device memory. assigned when being registerd

    def registrating(self, mem:DevMemory):
        self.mem = mem

    def get_gui(self):
        return self.gui

    def render(self):
        mem = self.mem
        size = mem.getNextId()
        active = mem.lifetime.to_numpy()[:size] != 0
        ph = mem.phase.to_numpy()[:size]
        if not self.display_fluid:
            active *= ph != FLUID
        pos = mem.curPos.to_numpy()[:size][active]
        pos /= self.window
        self.gui.circles(pos=pos,
                radius=4,
                color=PALETTE[ph[active]]
        )

    def draw_grid(self, grid):
        if not DRAW_GRID:
            return
        size = grid.grid_size
        x,y = grid.grid_shape
        w,h = self.window
        vert = [i*size/w for i in range(x)]
        hort = [i*size/h for i in range(y)]

        self.gui.lines(np.array([[m,0]for m in vert]), 
            np.array([[m,1]for m in vert]),
            color=self.bg_color, radius=1)

        self.gui.lines(np.array([[0,m]for m in hort]), 
            np.array([[1,m]for m in hort]), 
            color=self.bg_color, radius=1)

    def draw_neighbors(self, grid, x1):
        if not DRAW_NEIGHBOR:
            return
        mem = self.mem
        pos = []
        id = []
        for i in range(grid.n_neighbors[x1]):
            x2 = grid.neighbors[x1, i]
            id.append(x2)
            pos.append(mem.curPos[x2].value / self.window)
        if pos:
            print(len(id), sorted(id))
            self.gui.circles(np.array(pos), radius=3, color=0x00ff00)

    