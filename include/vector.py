import taichi as ti

def vec1(x=0.):
    return ti.Vector([x])

def vec2(x=0., y=0.):
    return ti.Vector([x, y])

def vec3(x=0., y=0., z=0.):
    return ti.Vector([x, y, z])

def vec4(x=0., y=0., z=0., w=0.):
    return ti.Vector([x, y, z, w])

def new_field(shape, dim=2, dtype=ti.f32):
    ''' Allocate Memory on Device '''
    if dim == 2:
        return ti.Vector.field(2, dtype=dtype, shape=shape)
    else:
        return ti.field(dtype, shape=shape)