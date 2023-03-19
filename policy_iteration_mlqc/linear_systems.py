from functools import reduce
from copy import copy
from time import time

import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are

from utility.matrixmath import vec, mat, mdot, matmul_lr, specrad, dlyap, dare, dare_gain
from quadtools import quadblock, quadstack, unquadblock, unquadstack


class LinearSystem:
    def __init__(self, A, B, C, a, Aa, b, Bb, c, Cc, Q, W):
        self.A = A
        self.B = B
        self.C = C
        self.a = a
        self.b = b
        self.c = c
        self.Aa = Aa
        self.Bb = Bb
        self.Cc = Cc
        self.Q = Q
        self.W = W
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]

    @property
    def data(self):
        return self.A, self.B, self.C, self.a, self.Aa, self.b, self.Bb, self.c, self.Cc, self.Q, self.W

    @property
    def dims(self):
        return self.n, self.m, self.p

    @property
    def AB(self):
        return np.block([self.A, self.B])

    @property
    def AC(self):
        return np.block([[self.A], [self.C]])


class LinearSystemControlled(LinearSystem):
    def __init__(self, system, K, L):
        super().__init__(*system.data)
        self.K = K
        self.L = L

        # Zeros matrices
        self.Zn = np.zeros([self.n, self.n])

    @property
    def BK(self):
        return self.B @ self.K

    @property
    def LC(self):
        return self.L @ self.C

    @property
    def F(self):
        return self.A + self.BK - self.LC

    @property
    def Phi_aug(self):
        return np.block([[self.A, self.BK],
                         [self.LC, self.F]])

    @property
    def AK(self):
        return self.A + self.BK

    @property
    def AL(self):
        return self.A - self.LC

    @property
    def IK(self):
        return np.block([[np.eye(self.n)], [self.K]])

    @property
    def IL(self):
        return np.block([np.eye(self.n), self.L])

    @property
    def QK(self):
        return matmul_lr(self.IK.T, self.Q)

    @property
    def WL(self):
        return matmul_lr(self.IL, self.W)

    @property
    def IK_aug(self):
        return sla.block_diag(np.eye(self.n), self.K)

    @property
    def IL_aug(self):
        return sla.block_diag(np.eye(self.n), self.L)

    @property
    def QK_aug(self):
        return matmul_lr(self.IK_aug.T, self.Q)

    @property
    def WL_aug(self):
        return matmul_lr(self.IL_aug, self.W)

    @property
    def linop1(self):
        # Closed-loop quadratic cost transition operator
        linop = np.kron(self.Phi_aug.T, self.Phi_aug.T)
        for i in range(self.a.size):
            PhiAa = np.block([[self.Aa[i], self.Zn],
                              [self.Zn, self.Zn]])
            linop += self.a[i]*np.kron(PhiAa.T, PhiAa.T)
        for i in range(self.b.size):
            PhiBb = np.block([[self.Zn, np.dot(self.Bb[i], self.K)],
                              [self.Zn, self.Zn]])
            linop += self.b[i]*np.kron(PhiBb.T, PhiBb.T)
        for i in range(self.c.size):
            PhiCc = np.block([[self.Zn, self.Zn],
                              [np.dot(self.L, self.Cc[i]), self.Zn]])
            linop += self.c[i]*np.kron(PhiCc.T, PhiCc.T)
        return linop

    @property
    def linop2(self):
        # Closed-loop second moment transition operator
        linop = np.kron(self.Phi_aug, self.Phi_aug)
        for i in range(self.a.size):
            PhiAa = np.block([[self.Aa[i], self.Zn],
                              [self.Zn, self.Zn]])
            linop += self.a[i]*np.kron(PhiAa, PhiAa)
        for i in range(self.b.size):
            PhiBb = np.block([[self.Zn, np.dot(self.Bb[i], self.K)],
                              [self.Zn, self.Zn]])
            linop += self.b[i]*np.kron(PhiBb, PhiBb)
        for i in range(self.c.size):
            PhiCc = np.block([[self.Zn, self.Zn],
                              [np.dot(self.L, self.Cc[i]), self.Zn]])
            linop += self.c[i]*np.kron(PhiCc, PhiCc)
        return linop

    @property
    def P_aug(self):
        linop = self.linop1
        r = specrad(linop)
        if r > 1:
            return np.full((2*self.n, 2*self.n), np.inf)
        else:
            I = np.eye((2*self.n)*(2*self.n))
            vQK = vec(self.QK_aug)
            return mat(la.solve(I - linop, vQK))

    @property
    def S_aug(self):
        linop = self.linop2
        r = specrad(linop)
        if r > 1:
            return np.full((2*self.n, 2*self.n), np.inf)
        else:
            I = np.eye((2*self.n)*(2*self.n))
            vWL = vec(self.WL_aug)
            return mat(la.solve(I - linop, vWL))

    @property
    def X(self):
        # NOTE: At the optimum, P_aug_xu + P_aug_uu = 0, but not for suboptimal policies.
        # NOTE: At the optimum, S_aug_xy - S_aug_yy = 0, but not for suboptimal policies.

        P_aug_xx, P_aug_uu, P_aug_xu, P_aug_ux = unquadblock(self.P_aug, self.n)
        S_aug_xx, S_aug_yy, S_aug_xy, S_aug_yx = unquadblock(self.S_aug, self.n)

        P = P_aug_xx + P_aug_xu + P_aug_ux + P_aug_uu
        Phat = P_aug_uu
        S = S_aug_xx - S_aug_xy - S_aug_yx + S_aug_yy
        Shat = S_aug_yy
        return quadstack(P, Phat, S, Shat)

    def qfun(self, X):
        P, Phat, S, Shat = unquadstack(X)

        # Control Q-function (G)
        # Get the noiseless part
        G = self.Q + matmul_lr(self.AB.T, P)
        # Add the noisy part in Guu block
        Gxx, Guu, Gxu, Gux = unquadblock(G, self.n)
        Guu += np.einsum('x,xji,jk,xkl->il', self.b, self.Bb, P, self.Bb)
        Guu += np.einsum('x,xji,jk,xkl->il', self.b, self.Bb, Phat, self.Bb)

        # Estimator Q-function (H)
        # Get the noiseless part in Hyy block
        H = self.W + matmul_lr(self.AC, S)
        # Add the noisy part
        Hxx, Hyy, Hxy, Hyx = unquadblock(H, self.n)
        Hyy += np.einsum('x,xij,jk,xlk->il', self.c, self.Cc, S, self.Cc)
        Hyy += np.einsum('x,xij,jk,xlk->il', self.c, self.Cc, Shat, self.Cc)

        # Compute gains for use in computing the Gxx, Hxx blocks
        K = -la.solve(Guu, Gux)  # Control gain  u = K*x
        L = la.solve(Hyy, Hyx).T  # Estimator gain  xhat = A*x + B*u + L*(y - C*xhat)

        LX2L = np.dot(L.T, np.dot(Phat, L))
        KX4K = np.dot(K, np.dot(Shat, K.T))

        Gxx += np.einsum('x,xji,jk,xkl->il', self.a, self.Aa, P, self.Aa)
        Gxx += np.einsum('x,xji,jk,xkl->il', self.a, self.Aa, Phat, self.Aa)
        Gxx += np.einsum('x,xji,jk,xkl->il', self.c, self.Cc, LX2L, self.Cc)

        Hxx += np.einsum('x,xij,jk,xlk->il', self.a, self.Aa, S, self.Aa)
        Hxx += np.einsum('x,xij,jk,xlk->il', self.a, self.Aa, Shat, self.Aa)
        Hxx += np.einsum('x,xij,jk,xlk->il', self.b, self.Bb, KX4K, self.Bb)

        # Put the blocks together
        G = quadblock(Gxx, Guu, Gxu, Gux)
        H = quadblock(Hxx, Hyy, Hxy, Hyx)
        return G, H

    def print_diagnostic(self, X, K, L, X_opt):
        P_opt, Phat_opt, S_opt, Shat_opt = unquadstack(X_opt)
        P, Phat, S, Shat = unquadstack(X)
        print("[" + ' '.join(["%+.6e" % val for val in K[0]]) + "] ", end='')
        print("[" + ' '.join(["%+.6e" % val for val in L.T[0]]) + "] ", end='')

        print("%.6e " % la.norm(P - P_opt), end='')
        print("%.6e " % la.norm(Phat - Phat_opt), end='')
        print("%.6e " % la.norm(S - S_opt), end='')
        print("%.6e " % la.norm(Shat - Shat_opt), end='')
        print('')

    def policy_evaluation(self):
        # Compute value function based on current policy
        # This is just a trivial wrapper around the X property,
        # which is a wrapper around the P_aug property,
        # which solves a generalized Lyapunov equation

        # It is trivially (if tediously) verified that
        # P == P_aug_xx + P_aug_xu + P_aug_ux + P_aug_uu
        # S == S_aug_xx - S_aug_xy - S_aug_yx + S_aug_yy
        # by expanding the relevant Lyapunov equations.
        return self.X

    def policy_improvement(self, X, return_qfun=False):
        # Compute state-action value matrices
        G, H = self.qfun(X)
        Gxx, Guu, Gxu, Gux = unquadblock(G, self.n)
        Hxx, Hyy, Hxy, Hyx = unquadblock(H, self.n)

        # Compute gains that improve based on current state-action value functions
        K = -la.solve(Guu, Gux)  # Control gain,  u = K @ x
        L = la.solve(Hyy, Hyx).T  # Estimator gain,  xhat = A @ x + B @ u + L @ (y - C @ xhat)

        if return_qfun:
            return K, L, G, H
        else:
            return K, L

    def ricc(self, X):
        # Riccati operator for multiplicative noise LQG
        # See W.L. de Koning, TAC 1992  https://ieeexplore.ieee.org/document/135491

        # Get gain and Q function
        K, L, G, H = self.policy_improvement(X, return_qfun=True)
        Gxx, Guu, Gxu, Gux = unquadblock(G, self.n)
        Hxx, Hyy, Hxy, Hyx = unquadblock(H, self.n)

        # Closed-loop system matrices
        ABK = self.A + np.dot(self.B, K)
        ALC = self.A - np.dot(L, self.C)

        # Form the RHS
        Z1 = np.dot(Gxu, la.solve(Guu, Gux))
        Z3 = np.dot(Hxy, la.solve(Hyy, Hyx))

        E = np.dot(ALC.T, np.dot(X[1], ALC))
        F = np.dot(ABK, np.dot(X[3], ABK.T))

        Y1 = Gxx - Z1
        Y2 = E + Z1
        Y3 = Hxx - Z3
        Y4 = F + Z3

        return quadstack(Y1, Y2, Y3, Y4)

    def policy_iteration(self, num_iters, convergence_tol=1e-12, show_diagnostic=False, save_hist=False, X_opt=None):
        if save_hist:
            X_hist = np.zeros([num_iters+1, 4, self.n, self.n])
            K_hist = np.zeros([num_iters+1, self.m, self.n])
            L_hist = np.zeros([num_iters+1, self.n, self.p])
            K_hist[0] = np.copy(self.K)
            L_hist[0] = np.copy(self.L)

        i = 0
        diff_mag = np.inf
        X = np.full(shape=(self.n, self.n), fill_value=np.inf)
        time_start = time()
        while diff_mag > convergence_tol:
            X_last = np.copy(X)

            if i >= num_iters:
                break

            X = self.policy_evaluation()
            K, L = self.policy_improvement(X)
            self.K, self.L = K, L

            diff_mag = la.norm(X - X_last)

            if save_hist:
                X_hist[i] = X
                K_hist[i+1] = K
                L_hist[i+1] = L

            if show_diagnostic:
                self.print_diagnostic(X, K, L, X_opt)

            i += 1

        time_end = time()
        time_elapsed = time_end - time_start

        # Truncate unused portion
        if save_hist:
            if i < num_iters:
                X_hist = X_hist[0:i]
                K_hist = K_hist[0:i]
                L_hist = L_hist[0:i]

        X = self.policy_evaluation()
        if save_hist:
            X_hist[-1] = X

        if save_hist:
            return dict(X=X, K=K, L=L, X_hist=X_hist, K_hist=K_hist, L_hist=L_hist, time_elapsed=time_elapsed)
        else:
            return dict(X=X, K=K, L=L, time_elapsed=time_elapsed)

    def value_iteration(self, num_iters, convergence_tol=1e-12, show_diagnostic=False, save_hist=False, X_opt=None):
        X = np.copy(self.X)
        if save_hist:
            X_hist = np.zeros([num_iters+1, 4, self.n, self.n])
            K_hist = np.zeros([num_iters+1, self.m, self.n])
            L_hist = np.zeros([num_iters+1, self.n, self.p])
            X_hist[0] = X

        i = 0
        diff_mag = np.inf
        time_start = time()
        while diff_mag > convergence_tol:
            X_last = np.copy(X)

            if i >= num_iters:
                break

            X = self.ricc(X)

            diff_mag = la.norm(X - X_last)

            if show_diagnostic or save_hist:
                K, L = self.policy_improvement(X)
                if save_hist:
                    X_hist[i+1] = X
                    K_hist[i] = K
                    L_hist[i] = L

                if show_diagnostic:
                    self.print_diagnostic(X, K, L, X_opt)

            i += 1

        time_end = time()
        time_elapsed = time_end - time_start

        # Truncate unused portion
        if save_hist:
            if i < num_iters:
                X_hist = X_hist[0:i]
                K_hist = K_hist[0:i]
                L_hist = L_hist[0:i]

        K, L = self.policy_improvement(X)
        if save_hist:
            K_hist[-1] = K
            L_hist[-1] = L
        self.K, self.L = K, L
        X = self.policy_evaluation()

        if save_hist:
            return dict(X=X, K=K, L=L, X_hist=X_hist, K_hist=K_hist, L_hist=L_hist, time_elapsed=time_elapsed)
        else:
            return dict(X=X, K=K, L=L, time_elapsed=time_elapsed)
