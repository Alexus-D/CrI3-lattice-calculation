import matplotlib.pyplot as plt
import lattice as lc
import numpy as np
import coordination_transformations as ct


def plot_lattice(lattice: lc.LatticeCrI3, ions=['Cr', 'I']):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Cr = np.array([node[1]['coords'] for node in lattice.graph.nodes.data()]).transpose()
    I = []
    for edge in lattice.graph.edges.data():
        coords = edge[2]['coords']
        for vec in coords:
            if np.array([vector != vec for vector in I]).any() or len(I) == 0:
                I.append(vec)
    I = np.array(I).transpose()

    if 'Cr' in ions:
        ax.scatter(Cr[0], Cr[2], s=10, c='r')
    if 'I' in ions:
        ax.scatter(I[0], I[2], s=10, c='b')
    plt.show()


def plot_lattice_3D(lattice: lc.LatticeCrI3, ions=['Cr', 'I']):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    Cr = np.array([node[1]['coords'] for node in lattice.graph.nodes.data()]).transpose()
    I = []
    for edge in lattice.graph.edges.data():
        coords = edge[2]['coords']
        for vec in coords:
            if np.array([vector != vec for vector in I]).any() or len(I) == 0:
                I.append(vec)
    I = np.array(I).transpose()

    if 'Cr' in ions:
        ax.scatter(Cr[0], Cr[1], Cr[2], s=10, c='r')
    if 'I' in ions:
        ax.scatter(I[0], I[1], I[2], s=10, c='b')
    plt.show()


def plot_spin_config(lattice):
    pass


def plot_free_energy(data):
    pass


def plot_free_energy_terms(data):
    pass


if __name__ == '__main__':
    N = 10
    lat = lc.LatticeCrI3(N)
    R = lat.length['cell'] * N / (2 * np.pi)
    lat.transform_coordinates(ct.bend, R)
    plot_lattice_3D(lat, ['Cr'])
    print("OK")
