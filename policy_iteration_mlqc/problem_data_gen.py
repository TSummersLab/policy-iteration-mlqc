"""
Problem data generation.
Generic outputs are:
:param n: Number of states, integer
:param m: Number of inputs, integer
:param p: Number of outputs, integer
:param A: System state matrix, n x n matrix
:param B: System control input matrix, n x m matrix
:param C: System output matrix, p x n matrix
:param a: State-multiplicative noise variances, na x 1 array
:param b: Input-multiplicative noise variances, nb x 1 array
:param c: Output-multiplicative noise variances, nc x 1 array
:param Aa: State-multiplicative noise direction matrices, na x n x n array
:param Bb: Input-multiplicative noise direction matrices, nb x n x m array
:param Cc: Output-multiplicative noise direction matrices, nc x p x n array
:param Q: State- and control-dependent quadratic cost, (n+m) x (n+m) matrix
:param W: Additive process and measurement noise covariance, (n+p) x (n+p) matrix
"""

import numpy as np
import numpy.random as npr
from numpy.random import default_rng
import numpy.linalg as la
import scipy.linalg as sla

from utility.matrixmath import specrad, mdot
from utility.pickle_io import pickle_import, pickle_export

from quadtools import quadblock, quadstack, unquadblock, unquadstack
from linear_systems import LinearSystem, LinearSystemControlled


def gen_rand_system(n=4, m=3, p=2, spectral_radius=0.9, additive_noise_scale=0.01, multiplicative_noise_scale=0.01,
                    na=1, nb=1, nc=1, rng=None):

    if rng is None:
        rng = default_rng()

    A = rng.normal(size=(n, n))
    A *= spectral_radius/specrad(A)
    B = rng.normal(size=(n, m))
    C = rng.normal(size=(p, n))

    # Direction matrices
    Aa = np.stack([rng.normal(size=(n, n)) for i in range(na)])
    Bb = np.stack([rng.normal(size=(n, m)) for i in range(nb)])
    Cc = np.stack([rng.normal(size=(p, n)) for i in range(nc)])

    # Variances
    a = multiplicative_noise_scale*rng.uniform(size=na)
    b = multiplicative_noise_scale*rng.uniform(size=nb)
    c = multiplicative_noise_scale*rng.uniform(size=nc)

    # Y = np.eye(p)  # Output penalty
    # Qxx = np.dot(C.T, np.dot(Y, C))  # State penalty
    Qxx = np.eye(n)
    Quu = np.eye(m)
    Wxx = additive_noise_scale*np.eye(n)  # State noise covariance
    Wyy = additive_noise_scale*np.eye(p)  # Output noise covariance
    Q = quadblock(Qxx, Quu)
    W = quadblock(Wxx, Wyy)

    return LinearSystem(A, B, C, a, Aa, b, Bb, c, Cc, Q, W)


def gen_pendulum_system(inverted=True, mass=10.0, damp=2.0, dt=0.1, noise_level=1.0):
    # Pendulum with forward Euler discretization
    # x[0] = angular position
    # x[1] = angular velocity

    n = 2
    m = 1
    p = 1

    if inverted:
        sign = 1
    else:
        sign = -1

    A = np.array([[1.0, dt],
                  [sign*mass*dt, 1.0-damp*dt]])
    B = np.array([[0],
                  [dt]])
    C = np.array([[1.0, 0.0]])

    # Direction matrices
    Aa1 = np.array([[0.0, 1.0],
                   [sign*mass, -damp]])
    Aa2 = np.array([[0.0, 0.0],
                   [sign*dt, 0.0]])
    Aa3 = np.array([[0.0, 0.0],
                   [0.0, -dt]])
    Aa = np.stack([Aa1, Aa2, Aa3])

    Bb1 = np.array([[0.0],
                    [1.0]])
    Bb2 = np.array([[1.0],
                    [0.0]])
    Bb = np.stack([Bb1, Bb2])

    Cc1 = np.array([[1.0, 0.0]])
    Cc2 = np.array([[0.0, 1.0]])
    Cc = np.stack([Cc1, Cc2])

    # Variances
    # a = np.array([0.0, 0.0, 0.0])
    # b = np.array([0.0, 0.0])
    # c = np.array([0.0, 0.0])

    a = noise_level*np.array([0.0, 0.0, 0.0])
    b = noise_level*np.array([1.0, 0.0])
    c = noise_level*np.array([0.0, 0.0])

    # a = np.array([0.001, 0.01, 0.01])
    # b = np.array([0.0001, 0.002])
    # c = np.array([0.001, 0.0001])

    # Y = np.eye(p)
    # Qxx = np.dot(C.T, np.dot(Y, C))
    Qxx = np.eye(n)
    Quu = np.eye(m)
    Wxx = 0.01*np.diag([0.0, 1.0])
    Wyy = 0.01*np.diag([0.1])
    Q = quadblock(Qxx, Quu)
    W = quadblock(Wxx, Wyy)

    # Use this to check if off-diagonal terms cause solver issues/errors
    # W = rng.rand(n+p, n+p)
    # W = 0.01*(W @ W.T)

    return LinearSystem(A, B, C, a, Aa, b, Bb, c, Cc, Q, W)


# def save_system(n, m, p, A, B, C, a, Aa, b, Bb, c, Cc, Q, W, dirname_out, filename_out):
#     variables = [n, m, p, A, B, C, a, Aa, b, Bb, c, Cc, Q, W]
#     variable_names = ['n', 'm', 'p', 'A', 'B', 'C', 'a', 'Aa', 'b', 'Bb', 'c', 'Cc', 'Q', 'W']
#     system_data = dict(((variable_name, variable) for variable_name, variable in zip(variable_names, variables)))
#     pickle_export(dirname_out, filename_out, system_data)
#
#
# def load_system(filename_in):
#     system_data = pickle_import(filename_in)
#     variable_names = ['n', 'm', 'p', 'A', 'B', 'C', 'a', 'Aa', 'b', 'Bb', 'c', 'Cc', 'Q', 'W']
#     return [system_data[variable] for variable in variable_names]


def setup_pendulum_system(noise_level=1.0):
    base_system = gen_pendulum_system(inverted=False, mass=10.0, damp=1.2, dt=0.1, noise_level=noise_level)
    K0 = np.array([[0, 0]])
    L0 = np.array([[0], [0]])

    system = LinearSystemControlled(base_system, K0, L0)

    return system, K0, L0


def setup_rand_system(noise_level, rng):
    A_specrad = rng.uniform()
    base_system = gen_rand_system(n=2, m=1, p=1, spectral_radius=A_specrad, multiplicative_noise_scale=0.0001, rng=rng)
    K0 = np.zeros([base_system.m, base_system.n])
    L0 = np.zeros([base_system.n, base_system.p])
    system = LinearSystemControlled(base_system, K0, L0)
    factor = 1.01
    target_specrad = 1.0
    while specrad(system.linop1)**0.5 < target_specrad:
        system.a *= factor
        system.b *= factor
        system.c *= factor
    system.a /= factor
    system.b /= factor
    system.c /= factor

    system.a *= noise_level
    system.b *= noise_level
    system.c *= noise_level

    print(specrad(system.linop1)**0.5)

    return system, K0, L0
