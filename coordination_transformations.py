import numpy as np


def rotate_vector(vector, axis, angle, degree=False):
    if degree:
        angle = angle * np.pi / 180
    if np.linalg.norm(vector) == 0:
        return vector
    e_rot = axis / np.linalg.norm(axis)
    vec_norm = np.dot(vector, e_rot)
    vec_plane = vector - vec_norm * e_rot
    vec_plane_norm = np.linalg.norm(vec_plane)
    e_p = vec_plane / vec_plane_norm
    e_o = np.cross(e_rot, e_p)
    vector_out = (np.cos(angle) * e_p + np.sin(angle) * e_o) * vec_plane_norm
    vector_out = vector_out + e_rot * vec_norm
    return vector_out


def bend(vector, R):
    angle = vector[0] / R
    vector[0] = R * np.sin(angle)
    vector[2] = R * np.cos(angle)
    return vector


if __name__ == '__main__':
    vec = [1, 1, 1]
    axis = [0, 0, 1]
    angle = -45
    print(rotate_vector(vec, axis, angle, True))
