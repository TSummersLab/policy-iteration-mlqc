import numpy as np

import numpy.linalg as la
import scipy.linalg as sla


def quadblock(Mxx, Myy, Mxy=None, Myx=None):
    if Mxy is None:
        return sla.block_diag(Mxx, Myy)
    else:
        if Myx is None:
            Myx = Mxy.T
        return np.block([[Mxx, Mxy],
                         [Myx, Myy]])


def unquadblock(M, n=None):
    if n is None:
        n = int(M.shape/2)
    return M[0:n, 0:n], M[n:, n:], M[0:n, n:], M[n:, 0:n]


def quadstack(M1, M2, M3, M4):
    return np.stack([M1, M2, M3, M4])


def unquadstack(M):
    return M[0], M[1], M[2], M[3]