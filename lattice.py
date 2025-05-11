import networkx as nx
import numpy as np
import coordination_transformations as ct


class LatticeCrI3:
    def __init__(self, N):
        self.Ku = 0.75 / 10**3 * 1.602176634 / 10**12  # effective anisotropy, erg/ion
        self.muB = 927.40100783 / 10**23  # erg/Gs
        self.magnetic_moment = 3.1 * self.muB  # erg / (Gs * ion)
        self.V0 = 1.1 / 10**3 * 1.602176634 / 10**12  # erg/angstrom**2
        self.angle = {'I-Cr-I': 87.12,
                      'Cr-I-Cr': 92.88,
                      'Cr-Cr-Cr': 120}
        for i, j in self.angle.items():
            self.angle[i] = j * np.pi / 180
        self.length = {'Cr-I': 2.73,
                       'I-Oxy': 1.58}
        self.length['I-I'] = 2. * self.length['Cr-I'] * np.sin(self.angle['I-Cr-I'] / 2)
        self.length['Cr-Cr'] = 2 * self.length['Cr-I'] * np.sin(self.angle['Cr-I-Cr'] / 2)
        self.angle['z-Cr-I'] = np.arccos(self.length['I-Oxy'] * 2. / self.length['I-I'])
        vector = np.array([0, 0, 0])
        axis_z = np.array([0, 0, 1])

        self.graph = nx.Graph()

        for i in range(N):
            self.graph.add_node(i * 4, coords=vector, spin=[0, 0, 1])  # Cr_1
            vector = vector + ct.rotate_vector([self.length['Cr-Cr'], 0, 0], axis_z, self.angle['Cr-Cr-Cr'] / 2)
            self.graph.add_node(i * 4 + 1, coords=vector, spin=[0, 0, 1])  # Cr2
            vector = vector + [self.length['Cr-Cr'], 0, 0]
            self.graph.add_node(i * 4 + 2, coords=vector, spin=[0, 0, 1])  # Cr3
            vector = vector + ct.rotate_vector([self.length['Cr-Cr'], 0, 0], axis_z, - self.angle['Cr-Cr-Cr'] / 2)
            self.graph.add_node(i * 4 + 3, coords=vector, spin=[0, 0, 1])  # Cr4
            vector = vector + [self.length['Cr-Cr'], 0, 0]
        self.length['cell'] = np.linalg.norm(vector) / N
        self.length['total'] = np.linalg.norm(vector)
        self.R = np.linalg.norm(vector) / (2 * np.pi)

        for i, node in enumerate(self.graph.nodes):
            if i == 0:
                continue
            vec_Cr_Cr = (self.graph.nodes()[i]['coords'] - self.graph.nodes()[i-1]['coords']) / 2
            vec_I_1 = vec_Cr_Cr + [0, 0, self.length['I-I'] / 2]
            vec_I_1 = ct.rotate_vector(vec_I_1, vec_Cr_Cr, self.angle['z-Cr-I'])
            vec_I_2 = ct.rotate_vector(vec_I_1, vec_Cr_Cr, 180, True)
            if i % 2:
                self.graph.add_edge(i-1, i,
                                    coords=[vec_I_1 + self.graph.nodes()[i]['coords'],
                                            vec_I_2 + self.graph.nodes()[i]['coords']],
                                    double=True)
            else:
                self.graph.add_edge(i-1, i,
                                    coords=[vec_I_1 + self.graph.nodes()[i]['coords'],
                                            vec_I_2 + self.graph.nodes()[i]['coords']],
                                    double=False)

        last_node = self.graph.number_of_nodes()-1
        vec_Cr_Cr = (vector - self.graph.nodes()[last_node]['coords']) / 2
        vec_I_1 = vec_Cr_Cr + [0, 0, self.length['I-I'] / 2]
        vec_I_1 = ct.rotate_vector(vec_I_1, vec_Cr_Cr, self.angle['z-Cr-I'])
        vec_I_2 = ct.rotate_vector(vec_I_1, vec_Cr_Cr, 180, True)
        self.graph.add_edge(last_node, 0,
                            coords=[vec_I_1 + self.graph.nodes()[last_node]['coords'],
                                    vec_I_2 + self.graph.nodes()[last_node]['coords']],
                            double=False)

    def transform_coordinates(self, transform_func, *args):
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes()[i]['coords'] = transform_func(self.graph.nodes()[i]['coords'], *args)
        for i in list(self.graph.edges):
            self.graph.edges()[i]['coords'] = [transform_func(j, *args) for j in self.graph.edges()[i]['coords']]

    def set_spins(self, func):
        for node in self.graph.nodes():
            rel_x = self.graph.nodes()[node]['coords'][0] / self.length['total']
            angle = func(rel_x)
            self.graph.nodes()[node]['spin'] = ct.rotate_vector(self.graph.nodes()[node]['spin'],
                                                                [0, 1, 0],
                                                                angle)

    def calc_free_energy_term(self, term_function, *args):
        return term_function(self, *args)

    def calc_free_energy(self, *term_functions):
        output = 0.
        for func in term_functions:
            output += self.calc_free_energy_term(func)
        return output


if __name__ == '__main__':
    lattice = LatticeCrI3(3)
    lattice.set_spins(lambda x: 2 * np.pi * x)
    print(lattice.graph.nodes.data())
