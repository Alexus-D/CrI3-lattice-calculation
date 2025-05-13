import numpy as np
import lattice as lc
import coordination_transformations as ct


unit_coefficient = {'eV': 6.24150965e11,
                    'meV': 6.24150965e14,
                    'erg': 1.}


def J1(epsilon):
    return (-1.52189 + 6.53823 * epsilon + 124.13611 * epsilon**2) * 1.602176634e-15  # in erg


def J2(epsilon):
    return (-1.52191 + 8.24436 * epsilon + 48.52742 * epsilon**2) * 1.602176634e-15  # in erg


def DMI(lattice: lc.LatticeCrI3, unit='meV'):
    graph = lattice.graph
    nodes = graph.nodes
    edges = graph.edges
    V0 = lattice.V0
    E = 0.
    for edge in edges:
        koef = 2 if edges[edge]['double'] else 1
        I1 = edges[edge]['coords'][0]
        I2 = edges[edge]['coords'][1]
        Cr1 = nodes[edge[0]]['coords']
        Cr2 = nodes[edge[1]]['coords']
        spin1 = nodes[edge[0]]['spin']
        spin2 = nodes[edge[1]]['spin']
        D1 = np.cross(Cr1 - I1, Cr2 - I1)
        D2 = np.cross(Cr1 - I2, Cr2 - I2)
        E += koef * V0 * np.dot(D1, np.cross(spin1, spin2))
        E += koef * V0 * np.dot(D2, np.cross(spin1, spin2))
    return E / graph.number_of_nodes() * unit_coefficient[unit]


def anisotropy(lattice: lc.LatticeCrI3, unit='meV'):
    graph = lattice.graph
    nodes = graph.nodes
    Ku = lattice.Ku
    E = 0.
    for node in nodes:
        spin = graph.nodes()[node]['spin']
        coords = graph.nodes()[node]['coords']
        norm_angle = np.arctan(coords[0] / coords[2]) if np.linalg.norm(coords) else 0.
        spin_angle = np.arctan(spin[0] / spin[2])
        E += -Ku * np.cos(norm_angle - spin_angle)**2
    return E / graph.number_of_nodes() * unit_coefficient[unit]


def magnetostatic(lattice: lc.LatticeCrI3, unit='meV'):
    mu0 = lattice.magnetic_moment
    graph = lattice.graph
    nodes = graph.nodes
    edges = graph.edges
    E = 0.
    for node in nodes:
        neighbours = graph[node]
        for neib in neighbours:
            W = 0.
            R = nodes[neib]['coords'] - nodes[node]['coords']
            R *= 1e-9  # angstrom to cm
            m_neib = - np.array(nodes[neib]['spin']) * mu0
            m_node = - np.array(nodes[node]['spin']) * mu0
            W += np.dot(m_node, m_neib) * np.linalg.norm(R)**2
            W += - 3 * np.dot(m_node, R) * np.dot(m_neib, R)
            W /= np.linalg.norm(R)**5
            koef = 1 if edges[node, neib]['double'] else 0.5
            E += W * koef
    return E / graph.number_of_nodes() / 2 * unit_coefficient[unit]


def isotropic_exchange(lattice: lc.LatticeCrI3, unit='meV'):
    graph = lattice.graph
    nodes = graph.nodes().data()
    edges = graph.edges
    E = 0.
    for node in nodes:
        neighbours = graph[node[0]]
        for neib in neighbours:
            epsilon = (np.linalg.norm(node[1]['coords'] - graph.nodes[neib]['coords']) - lattice.length['Cr-Cr'])\
                      / lattice.length['Cr-Cr']
            if edges[node[0], neib]['double']:
                J = J2(epsilon)
            else:
                J = J1(epsilon) / 2
            E += J * np.dot(node[1]['spin'], graph.nodes[neib]['spin'])
    return E / graph.number_of_nodes() * unit_coefficient[unit]


if __name__ == '__main__':
    lat = lc.LatticeCrI3(3)
    lat.set_spins(lambda x: x * 2 * np.pi)
    E = lat.calc_free_energy_term(anisotropy, 'meV')
    print(E)
