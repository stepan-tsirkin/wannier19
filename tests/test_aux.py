"""test auxilary functions"""

from random import shuffle
import numpy as np
from pytest import approx
from wannierberri.formula.covariant import _spin_velocity_einsum_opt
from wannierberri.__utility import points_to_mp_grid, vectorize


def test_spin_velocity_einsum_opt():

    nw = 5
    nk = 6
    for i in range(10):
        A = np.random.random((nk, nw, nw, 3))
        B = np.random.random((nk, nw, nw, 3))
        C = np.random.random((nk, nw, nw, 3, 3))
        C1 = np.copy(C)
        _spin_velocity_einsum_opt(C, A, B)
        # Optimized version of C += np.einsum('knls,klma->knmas', A, B). Used in shc_B_H.
        C1 += np.einsum('knls,klma->knmas', A, B)
        assert C1 == approx(C)


def test_vectorized_eigh():
    nw = 5
    nk = 6
    for i in range(10):
        H = np.random.random((nk, nw, nw)) + 1j * np.random.random((nk, nw, nw))
        H = H + H.transpose(0, 2, 1).conj()
        EV = vectorize(np.linalg.eigh, H)
        for h, ev in zip(H, EV):
            e1, v1 = ev
            e2, v2 = np.linalg.eigh(h)
            assert e1 == approx(e2)


def test_vectorized_matrix_prod():
    nw = 5
    nk = 6
    for i in range(10):
        A = np.random.random((nk, nw, nw)) + 1j * np.random.random((nk, nw, nw))
        B = np.random.random((nk, nw, nw)) + 1j * np.random.random((nk, nw, nw))
        C = vectorize(np.dot, A, B)
        for a, b, c in zip(A, B, C):
            c1 = np.dot(a, b)
            assert c == approx(c1)

def test_points_to_grid():
    for grid in (2,3,4), (1,1,1), (3,3,3):
        points = [(i/grid[0],j/grid[1],k/grid[2]) for i in range(grid[0]) for j in range(grid[1]) for k in range(grid[2])]
        print("points =", points)
        shuffle(points)
        print("points shuffled =", points)
        grid_new = points_to_mp_grid(points)
        assert grid == grid_new, f"failed for grid {grid} got {grid_new}"