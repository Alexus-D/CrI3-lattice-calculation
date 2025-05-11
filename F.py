import numpy as np
import lattice as lc


def J1(cr1, cr2):
    distance = np.linalg.norm(cr2 - cr1)
    return -1.52189 + 6.53823 * distance + 124.13611 * distance**2


def J2(cr1, cr2):
    distance = np.linalg.norm(cr2 - cr1)
    return -1.52191 + 8.24436 * distance + 48.52742 * distance**2

def DMI(lattice: lc.LatticeCrI3):
    graph = lattice.graph
    nodes = graph.nodes
    edges = graph.edges
    V0 = lattice.V0
    E  = 0.
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
    return E / graph.number_of_nodes()


def anisotropy(lattice: lc.LatticeCrI3):
    graph = lattice.graph
    nodes = graph.nodes
    Ku = lattice.Ku
    E = 0.
    for node in nodes:
        spin = graph.nodes()[node]['spin']
        coords = graph.nodes()[node]['coords']
        norm_angle = np.arctan(coords[0] / coords[2])
        spin_angle = spin[0] / spin[2]
        E += -Ku * np.cos(norm_angle - spin_angle)**2
    return E / graph.number_of_nodes()


def magnetostatic(lattice: lc.LatticeCrI3):
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
            m_neib = - np.array(nodes[neib]['spin']) * mu0
            m_node = - np.array(nodes[node]['spin']) * mu0
            W += np.dot(m_node, m_neib) * np.linalg.norm(R)**2
            W += - 3 * np.dot(m_node, R) * np.dot(m_neib, R)
            W /= np.linalg.norm(R)**5
            koef = 1 if edges[node, neib]['double'] else 0.5
            E += W  * koef
    return E / graph.number_of_nodes() / 2


def isotropic_exchange(lattice: lc.LatticeCrI3):
    graph = lattice.graph
    nodes = graph.nodes().data()
    edges = graph.edges
    E = 0.
    for node in nodes:
        neighbours = graph[node[0]]
        for neib in neighbours:
            if edges[node[0], neib]['double']:
                J = 2 * J2(node[1]['coords'], graph.nodes[neib]['coords'])
            else:
                J = 2 * J1(node[1]['coords'], graph.nodes[neib]['coords'])
            E += - J * np.dot(node[1]['spin'], graph.nodes[neib]['spin'])
    return E / graph.number_of_nodes() / 2


if __name__ == '__main__':
    lat = lc.LatticeCrI3(3)
    E = lat.calc_free_energy_term(magnetostatic)
    print(E)
