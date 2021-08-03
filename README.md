# taichi-PBF
 Position Based Fluid Simulation in Taichi

# Features
1. Fluid Sim (water pool)
2. Smoke/Gas Sim (smoke plume rising in open area)

# TODO
1. ~~Values like poly6 and spikyGradient are highly reusable and should be cached before each update~~
2. So far I store the entire grid with a giant matrix. However, many of the cells doesn't contain any particles at all, which is a huge waste of memory. A better approach would be using a dictionary and store the hash value of cell which has particles exists
3. After adding certain number of particles, the simulation will break due to the data structure capacity. There are two approaches to fix this:
- Reallocate and move data to a bigger memory when the is reached. Huge overheads when this operation happens.
- Recycle dead particles. Keep an index pool, fetch one when new particle is created, and return one when a particle dies.
