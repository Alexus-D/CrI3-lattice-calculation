import numpy as np
import lattice as lc
import coordination_transformations as ct
import F


cell = 11.869818272463824
uniform_exchange = []
for N in range(40, 60):
    length = N * cell
    R = length / (2 * np.pi)
    lat = lc.LatticeCrI3(N)
    lat.transform_coordinates(ct.bend, R)
    lat.set_spins(lambda x: 0)
    # lat.set_spins(lambda x: x * 2 * np.pi)
    # E = lat.calc_free_energy_term(F.isotropic_exchange)
    # E = lat.calc_free_energy_term(F.anisotropy)
    # E = lat.calc_free_energy_term(F.DMI)
    E = lat.calc_free_energy_term(F.magnetostatic)
    uniform_exchange.append([R, E])

uniform_exchange = np.array(uniform_exchange)
np.savetxt(f'Magnetostatic-from-radius-curvature(uniform-spins).txt', uniform_exchange)
