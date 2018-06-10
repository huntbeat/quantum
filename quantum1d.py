# MATH056 - Hyong Hark Lee, Minseo Park, David Xu
# Quantum walk simulation
# 1D representation

import numpy as np

runs = 20 
locations = runs * 2 + 5
spin_dimension = 1 * 2
shape = (locations,spin_dimension)

up = 0
down = 1

location_spin = np.zeros(shape, dtype=np.complex_)

location_spin[locations/2,up] = 1;
location_spin[locations/2,down] = 1j;

sig = np.sum(location_spin[locations/2,:])

location_spin[locations/2,up] /= sig;
location_spin[locations/2,down] /= sig;

for run in range(runs):
    update = np.zeros(shape, dtype=np.complex_)
    for loc in range(locations):
        if loc < locations - 1 and loc > 1:
            update[loc+1,up] += location_spin[loc,up]
            update[loc-1,down] += location_spin[loc,up]
            update[loc+1,up] += location_spin[loc,down]
            update[loc-1,down] -= location_spin[loc,down]
    location_spin = update

location_spin = np.multiply(location_spin, np.conjugate(location_spin))
location_spin = np.divide(  location_spin,
                            np.sqrt(2) ** runs 
                                )
location_spin = np.sum(location_spin, axis=1)

print(location_spin)

