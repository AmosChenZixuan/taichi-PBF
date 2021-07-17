import taichi as ti
from include import *
from src import Simulation, Renderer
from time import perf_counter as clock
ti.init(arch=ARCH)


def timeit(c, what):
    print(what, clock() - c)
    return c

backend  = Simulation()
frontend = Renderer()
backend.register_externals(frontend)
gui = frontend.get_gui()

backend.reset()

while gui.running:
    backend.tick(1)
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'p':
            backend.paused = not backend.paused
        elif e.key == 'r':
            backend.reset()
        elif e.key == 'e':
            frontend.display_fluid = not frontend.display_fluid
        elif e.key == gui.SPACE:
            backend.emit_smoke()
    if False:
        if not (backend.tick() % 5):
            backend.emit_smoke()

    if gui.is_pressed(ti.GUI.RMB):
        frontend.mouse_pos = gui.get_cursor_pos()
        frontend.attract = 1
    elif gui.is_pressed(ti.GUI.LMB):
        frontend.mouse_pos = gui.get_cursor_pos()
        frontend.attract = -1
    else:
        backend.attract = 0

    if DEBUG_MODE:
        print(f'===={backend.tick()}====')
        timer = clock()
        # step
        backend.step()
        timer = timeit(timer, 'step')
        # render
        frontend.render()
        frontend.draw_grid(backend.grid)
        frontend.draw_neighbors(backend.grid, 10)
        timer = timeit(timer, 'render')
        # display
        gui.show()
        timer = timeit(timer, 'show')
    else:
        backend.step()
        frontend.render()
        gui.show()

