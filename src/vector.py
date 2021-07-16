import taichi as ti

def vec3(x=0.):
    return ti.Vector([x])

def vec2(x=0., y=0.):
    return ti.Vector([x, y])

def vec3(x=0., y=0., z=0.):
    return ti.Vector([x, y, z])

def vec4(x=0., y=0., z=0., w=0.):
    return ti.Vector([x, y, z, w])
