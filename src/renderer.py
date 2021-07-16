import taichi as ti
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