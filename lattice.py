import networkx as nx
import numpy as np


class LatticeCrI3:
    def __init__(self, N):
        self.angle = {'I-Cr-I': 87.12,
                      'Cr-I-Cr': 92.88,
                      'Cr-Cr-Cr': 120}
        for i, j in self.angle.items():
            self.angle[i] = j * np.pi / 180
        self.length = {'Cr-I': 2.73,
                       'I-Oxy': 1.58}
        self.length['I-I'] = 2. * self.length['Cr-I'] * np.sin(self.angle['I-Cr-I'] / 2)
        self.angle['z-Cr-I'] = np.arccos(self.length['I-Oxy'] * 2. / self.length['I-I'])
        self.graph = nx.Graph()

    def transform_coordinates(self, transform_func, *args):
        pass

    def calc_free_energy_term(self, term_function):
        pass

    def calc_free_energy(self, *term_functions):
        pass


if __name__ == '__main__':
    lattice = LatticeCrI3(1)
    print(lattice)
