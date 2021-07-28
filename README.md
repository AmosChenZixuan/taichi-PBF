# taichi-PBF
 Position Based Fluid Simulation in Taichi

# Features
1. Fluid Sim (water pool)
2. Smoke/Gas Sim (smoke plume rising in open area)

# TODO
1. Values like particle distances are highly reusable and should be cached before each update
2. So far I store the entire grid with a giant matrix. However, many of the cells doesn't contain any particles at all, which is a huge waste of memory. A better approach would be using a dictionary and store the hash value of cell which has particles exists
