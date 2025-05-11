import numpy as np
import lattice as lc
import coordination_transformations as ct
import F
from scipy.ndimage import gaussian_filter



cell = 11.869818272463824
energy = []
r = []
for N in range(4, 50):
    length = N * cell
    R = length / (2 * np.pi)
    lat = lc.LatticeCrI3(N)
    lat.transform_coordinates(ct.bend, R)
    # lat.set_spins(lambda x: 0)
    lat.set_spins(lambda x: x * 2 * np.pi)
    # E = lat.calc_free_energy_term(F.isotropic_exchange)
    # E = lat.calc_free_energy_term(F.anisotropy)
    E = lat.calc_free_energy_term(F.DMI)
    # E = lat.calc_free_energy_term(F.magnetostatic)
    r.append(R)
    energy.append(E)

energy = np.array(energy)
energy = gaussian_filter(energy, sigma=3)

output = np.array([r, energy]).transpose()
np.savetxt(f'DMI-from-radius-curvature(hedgehog).txt', output)
