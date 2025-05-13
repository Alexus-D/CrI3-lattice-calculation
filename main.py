import numpy as np
import lattice as lc
import coordination_transformations as ct
import F
from scipy.ndimage import gaussian_filter



cell = 11.869818272463824
file_names = {'anisotropy-from-radius-curvature': F.anisotropy,
              'DMI-from-radius-curvature': F.DMI,
              'Magnetostatic-from-radius-curvature': F.magnetostatic,
              'uniform-exchange-from-radius-curvature': F.isotropic_exchange}
spin_configs = {'uniform-spins': lambda x: 0, 'hedgehog': lambda x: x * 2 * np.pi}

for file_name in file_names.items():
    for config in spin_configs.items():
        energy = []
        curvature = []
        for N in range(10, 50):
            length = N * cell
            R = length / (2 * np.pi)
            lat = lc.LatticeCrI3(N)
            lat.transform_coordinates(ct.bend, R)
            lat.set_spins(config[1])
            curvature.append(1 / R)
            energy.append(lat.calc_free_energy_term(file_name[1], 'meV'))
        energy = np.array(energy)
        energy = gaussian_filter(energy, sigma=4)
        output = np.array([curvature, energy]).transpose()
        np.savetxt(f"Images\\{file_name[0]}({config[0]}).txt", output)
