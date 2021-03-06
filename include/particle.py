from enum import Enum, unique
from include.vector import *
import numpy as np

'''
0. Fluid particle: Basic unit in PBF
1. Gas Particle: Basically Fluid, participate in computation
2. Smoke Particle: Injected visually, advect passively. No computation
'''
FLUID = 0
GAS   = 1
SMOKE = 2
SOLID = 3
CLOTH = 4

@unique
class Phase(Enum):
    fluid = FLUID
    gas   = GAS
    smoke = SMOKE
    solid = SOLID
    cloth = CLOTH

# particle colors to be rendered
PALETTE = np.array([0x328ac1, 0xff1c8a, 0x959595, 0x00ffff, 0xffff00])

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


'''
Constraint Groups

STABILIZATION: Push particles out of invalid state and fix their velocity
STANDARD     : Appiled to a fixed number of particles.                  e.g. Fluid/Gas, Distance
CONTACT      : Appiled to some particles, cleared after each iteration. e.g. Collision
SHAPE        : Appiled to vertices of a solid mesh. Different substep   e.g. Shape Matching
'''

STABILIZATION = 0
STANDARD      = 1
CONTACT       = 2
SHAPE         = 3

def newConstraintGroups():
    return [[] for _ in range(4)]