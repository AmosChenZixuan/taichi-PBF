from enum import Enum, unique
from src.vector import vec2


'''
0. Fluid particle: Basic unit in PBF
1. Gas Particle: Basically Fluid, participate in computation
2. Smoke Particle: Injected visually, advect passively. No computation
'''
FLUID = 0
GAS   = 1
SMOKE = 2

@unique
class Phase(Enum):
    fluid = FLUID
    gas   = GAS
    smoke = SMOKE

class Particle:
    ''' 
        Stay on the host memory
        Store the initial state of a particle
    '''
    def __init__(self, id, pos, vel=vec2(), acc=vec2(), mass=1., phase=FLUID, lifetime=-1):
        self.id       = id
        self.pos      = pos
        self.vel      = vel
        self.acc      = acc
        self.mass     = mass
        self.phase    = phase
        self.lifetime = lifetime

        # self.movable  = True
        # self.inv_m    = 0.
        # if mass > 0:
        #     self.inv_m   = 1/mass
        # else:
        #     self.movable = False     
