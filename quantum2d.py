# MATH056 - Hyong Hark Lee, Minseo Park, David Xu
# Quantum walk simulation
# 2D representation

import numpy as np

runs = 50 
location1 = runs * 2 + 5 # x axis
location2 = runs * 2 + 5 # y axis
spin_dimension = 2 * 2
shape = (location1,location2,spin_dimension)

left = 0
right = 2
up = 1
down = 3

location_spin = np.zeros(shape, dtype=np.complex_)

location_spin[location1/2,location2/2,up] = 0.5;
location_spin[location1/2,location2/2,down] = 0.5j;
location_spin[location1/2,location2/2,right] = 0.5;
location_spin[location1/2,location2/2,left] = 0.5j;

sig = np.multiply(location_spin[location1/2,location2/2,:], 
                    np.conjugate(location_spin[location1/2,location2/2,:]))
sig = np.sum(sig)

location_spin[location1/2,location2/2,up] /= sig;
location_spin[location1/2,location2/2,down] /= sig;
location_spin[location1/2,location2/2,right] /= sig;
location_spin[location1/2,location2/2,left] /= sig;

for run in range(runs):
    update = np.zeros(shape, dtype=np.complex_)
    for loc1 in range(location1):
        if loc1 < location1 - 1 and loc1 > 1:
          for loc2 in range(location2):
              if loc2 < location2 - 1 and loc2 > 1:
                  update[loc1+1,loc2,left] += ( location_spin[loc1,loc2,left] * -1 +
                                                location_spin[loc1,loc2,up] +
                                                location_spin[loc1,loc2,right] +
                                                location_spin[loc1,loc2,down] )
                  update[loc1-1,loc2,up] += ( location_spin[loc1,loc2,left] +
                                                location_spin[loc1,loc2,up] * -1 +
                                                location_spin[loc1,loc2,right] +
                                                location_spin[loc1,loc2,down] )
                  update[loc1,loc2+1,right] += ( location_spin[loc1,loc2,left] +
                                                location_spin[loc1,loc2,up] +
                                                location_spin[loc1,loc2,right] * -1 +
                                                location_spin[loc1,loc2,down] )
                  update[loc1,loc2-1,down] += ( location_spin[loc1,loc2,left] +
                                                location_spin[loc1,loc2,up] +
                                                location_spin[loc1,loc2,right] + 
                                                location_spin[loc1,loc2,down] * -1 )
    location_spin = update

location_spin = np.multiply(location_spin, np.conjugate(location_spin))
location_spin = np.divide(  location_spin,
                            4.0 ** runs 
                                )
location_spin = np.sum(location_spin, axis=2)

print(location1/2)
print(np.argmax(location_spin,axis=0))
print(np.argmax(location_spin,axis=1))
print(location_spin)
