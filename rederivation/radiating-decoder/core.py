"""Core numerics for the radiating-decoder re-derivation (agent radiating-decoder).

All momentum integrals are done on uniform grids (trapezoid == plain sum for integrands that
vanish at the window edges) in LOG space with complex log-sum-exp contractions, so that the huge
factors 2k sinh(2 pi k), Gamma(...) never over/underflow.

Disk rules (00_common.md):  region weight  dmu(k) exp(-s_b * l * k^2),  dmu = 2k sinh(2pi k) dk,
chord g_w(a,b) = Gamma(w +- ia +- ib)/(2 pi^2 Gamma(2w)).
"""
import numpy as np
from scipy.special import loggamma

LOG2PI2 = np.log(2 * np.pi ** 2)


def logmu(k):
    k = np.asarray(k, float)
    return np.log(2 * k) + 2 * np.pi * k + np.log1p(-np.exp(-4 * np.pi * k)) - np.log(2.0)


def logg(w, a, b):
    """log g_w(a,b) for real a,b (broadcast)."""
    s = a + b
    d = a - b
    return (2 * loggamma(w + 1j * s).real + 2 * loggamma(w + 1j * d).real
            - LOG2PI2 - loggamma(2 * w).real)


def lse(T, axis):
    """complex log-sum-exp along axis."""
    m = np.max(T.real, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(T - m), axis=axis)) + np.squeeze(m, axis=axis)
    return out


def physical_map(k1, tsp, m):
    k0 = np.sqrt(k1 ** 2 - 2 * m * k1 / np.tan(tsp) - m ** 2)
    A = np.arctan2(k0 + k1, m)
    B = np.arctan2(k0 - k1, m)
    t0 = (np.pi - A - B) / k0
    bM = t0 + tsp / k1
    C = np.pi / bM
    kap = k1 / C
    return dict(C=C, kap=kap, ts=tsp / kap, k0=k0, sb=1 / (2 * C))


def grid(lo, hi, h):
    lo = max(lo, h / 2)
    n = int(np.ceil((hi - lo) / h)) + 1
    x = lo + h * np.arange(n)
    return x, h


class Setup:
    """Holds parameters and grids. All lengths in bare u-units."""

    def __init__(self, C, ts, Delta=2.0, omega=0.2, hV=1.0, eps=0.05, De=1.0, ee=0.0125,
                 grids=None):
        self.C, self.ts, self.sb = C, ts, 1 / (2 * C)
        self.Delta, self.omega, self.hV, self.eps = Delta, omega, hV, eps
        self.De, self.ee = De, ee
        self.Dchi = Delta + omega / 2
        self.grids = grids  # dict name -> (x, h)
        self.tD = self.sb * (2 * np.pi - 2 * ts)

    def weights(self, a):
        Da = self.Delta + a * self.omega
        h, Dc = self.hV, self.Dchi
        return dict(psiV=(Da + h - Dc) / 2, psichi=(Da + Dc - h) / 2, Vchi=(h + Dc - Da) / 2)

    # ---- ket triangle --------------------------------------------------------------
    def logcap(self, a, y, Igrid):
        """log of  cap_a(I) = int dmu(A) e^{-sb (ts - eps/2 + i y) A^2} g_{psiV}(A, I)"""
        A, hA = self.grids['A']
        w = self.weights(a)['psiV']
        lA = logmu(A) - self.sb * (self.ts - self.eps / 2 + 1j * y) * A ** 2 + np.log(hA)
        T = lA[:, None] + logg(w, A[:, None], Igrid[None, :])
        return lse(T, axis=0)

    def logItri(self, a, y, Dgrid, Mgrid, chunk=8):
        """log I_a(D, M): ket triangle with cap A and inner region I integrated.
        I_a(D,M) = int dmu(I) cap_a(I) g_{psichi}(I,D) g_{Vchi}(I,M)"""
        I, hI = self.grids['I']
        w = self.weights(a)
        lc = self.logcap(a, y, I) + logmu(I) + np.log(hI)
        L1 = logg(w['psichi'], Dgrid[:, None], I[None, :])  # D x I
        L2 = logg(w['Vchi'], I[:, None], Mgrid[None, :])  # I x M
        out = np.empty((len(Dgrid), len(Mgrid)), complex)
        for i0 in range(0, len(Dgrid), chunk):
            T = L1[i0:i0 + chunk, :, None] + lc[None, :, None] + L2[None, :, :]
            out[i0:i0 + chunk] = lse(T, axis=1)
        return out

    # ---- one-sided -------------------------------------------------------------------
    def logWe(self, Bgrid, emission=True, De=None, ee=None):
        """log weight of the middle region B in the one-sided register.
        no emission: e^{-sb eps B^2};  emission returning on same boundary:
        e^{-sb (eps-2ee) B^2} int dmu(E) e^{-2 sb ee E^2} g_{De}(B,E)"""
        if not emission:
            return -self.sb * self.eps * Bgrid ** 2
        De = self.De if De is None else De
        ee = self.ee if ee is None else ee
        E, hE = self.grids['E']
        lE = logmu(E) - 2 * self.sb * ee * E ** 2 + np.log(hE)
        T = lE[None, :] + logg(De, Bgrid[:, None], E[None, :])
        return -self.sb * (self.eps - 2 * ee) * Bgrid ** 2 + lse(T, axis=1)

    def onesided(self, y, emission=True, De=None, ee=None, return_parts=False):
        D, hD = self.grids['D']
        B, hB = self.grids['B']
        lD = logmu(D) - self.tD * D ** 2 + np.log(hD)
        lB = logmu(B) + self.logWe(B, emission, De, ee) + np.log(hB)
        LI = [self.logItri(a, y, D, B) for a in (0, 1)]
        lgc = logg(self.Dchi, D[:, None], B[None, :])
        base = lD[:, None] + lB[None, :] - lgc
        F = {}
        for a in (0, 1):
            for b in (0, 1):
                T = base + LI[a] + np.conj(LI[b])
                F[(a, b)] = lse(T.ravel(), axis=0)
        return F

    # ---- decoded -----------------------------------------------------------------------
    def logF_chi(self, Sgrid):
        D, hD = self.grids['D']
        lD = logmu(D) - self.tD * D ** 2 + np.log(hD)
        T = lD[:, None] + logg(self.Dchi, D[:, None], Sgrid[None, :])
        return lse(T, axis=0)

    def logS(self, a, y, Sgrid, LI=None):
        D, hD = self.grids['D']
        lD = logmu(D) - self.tD * D ** 2 + np.log(hD)
        if LI is None:
            LI = self.logItri(a, y, D, Sgrid)
        return lse(lD[:, None] + LI, axis=0)

    def decoded0(self, y, Bweight=None):
        """Y^0 (no emission threading). Bweight: optional log weight replacing -sb eps s^2
        (used for the hybrid 'decoded, D separate, middle region shared with the one-sided
        emission bubble')."""
        S, hS = self.grids['B']
        lS = logmu(S) + np.log(hS) - self.logF_chi(S)
        lS = lS + (-self.sb * self.eps * S ** 2 if Bweight is None else Bweight(S))
        LS = [self.logS(a, y, S) for a in (0, 1)]
        return {(a, b): lse(lS + LS[a] + np.conj(LS[b]), axis=0) for a in (0, 1) for b in (0, 1)}

    def decoded_e(self, y, De=None, ee=None, share_D=False):
        """Decoded register with threading emission lines (m -> 0 replica continuation):
        rho_ab = int dmu(E) e^{-2 sb ee E^2} K_a(E) conj(K_b(E)) / P(E)
        K_a(E) = int dmu(s) e^{-sb (ts + eps/2 - 2ee - i y) s^2} g_De(s,E) S_a(s)
        P(E)   = int dmu(s) e^{-sb (2ts - 2ee) s^2} g_De(s,E) F_chi(s)
        share_D=True: hybrid in which ket and bra share one D region (amputated by the
        geometric mean sqrt(g_chi(D,s_k) g_chi(D,s_b))) -- diagnostic only."""
        De = self.De if De is None else De
        ee = self.ee if ee is None else ee
        S, hS = self.grids['S']
        E, hE = self.grids['E']
        sb = self.sb
        lge = logg(De, S[:, None], E[None, :])  # S x E
        lFc = self.logF_chi(S)
        lP = lse((logmu(S) + np.log(hS) - sb * (2 * self.ts - 2 * ee) * S ** 2 + lFc)[:, None] + lge,
                 axis=0)
        lE = logmu(E) + np.log(hE) - 2 * sb * ee * E ** 2 - lP
        lws = logmu(S) + np.log(hS) - sb * (self.ts + self.eps / 2 - 2 * ee - 1j * y) * S ** 2
        if not share_D:
            LK = []
            for a in (0, 1):
                lSa = self.logS(a, y, S)
                LK.append(lse((lws + lSa)[:, None] + lge, axis=0))
            return {(a, b): lse(lE + LK[a] + np.conj(LK[b]), axis=0) for a in (0, 1) for b in (0, 1)}
        # hybrid with shared D
        D, hD = self.grids['D']
        lD = logmu(D) - self.tD * D ** 2 + np.log(hD)
        lgc = logg(self.Dchi, D[:, None], S[None, :])  # D x S
        LI = [self.logItri(a, y, D, S) - 0.5 * lgc for a in (0, 1)]
        # J_a(D,E) = sum_s ws(s) ge(s,E) I_a(D,s)/sqrt(g(D,s))
        LJ = []
        for a in (0, 1):
            T = (lws[None, :] + LI[a])[:, :, None] + lge[None, :, :]
            LJ.append(lse(T, axis=1))  # D x E
        out = {}
        for a in (0, 1):
            for b in (0, 1):
                T = lD[:, None] + lE[None, :] + LJ[a] + np.conj(LJ[b])
                out[(a, b)] = lse(T.ravel(), axis=0)
        return out


def readout(F, omega):
    lv = F[(0, 1)] - 0.5 * (F[(0, 0)] + F[(1, 1)])
    V = np.exp(lv)
    return np.angle(V) / omega, 1 - abs(V)


def classical_reading(tau, y, eps):
    """fixed-geometry (conformal block) reading, see report: arg[s12 s13], s_ij = 2 sin((u_j-u_i)/2)"""
    s12 = np.sin((tau - eps / 2 + 1j * y) / 2)
    s13 = np.sin((tau + eps / 2 + 1j * y) / 2)
    return np.angle(s12) + np.angle(s13)


def classical_block(tau, y, eps, Delta, omega, h, Dchi=None):
    """full classical visibility from the 1d conformal block (returns reading, 1-|V|)."""
    from mpmath import hyp2f1
    if Dchi is None:
        Dchi = Delta + omega / 2
    u1, u2, u3, u4 = -tau, 1j * y - eps / 2, 1j * y + eps / 2, tau
    s = lambda a, b: 2 * np.sin((b - a) / 2)
    s12, s13, s14, s24, s34 = s(u1, u2), s(u1, u3), s(u1, u4), s(u2, u4), s(u3, u4)
    chi = (s12 * s34 / (s13 * s24)).real
    G = {}
    for a in (0, 1):
        for b in (0, 1):
            Da, Db = Delta + a * omega, Delta + b * omega
            d12, d34 = Da - h, h - Db
            pref = (s12 ** (-Da - h) * s34 ** (-h - Db) * (s24 / s14) ** d12 * (s14 / s13) ** d34)
            G[(a, b)] = pref * chi ** Dchi * complex(hyp2f1(Dchi - d12, Dchi + d34, 2 * Dchi, chi))
    V = G[(0, 1)] / np.sqrt(G[(0, 0)] * G[(1, 1)])
    return np.angle(V) / omega, 1 - abs(V)


# ------------------------------------------------------------------------------------------
# Efficient driver: compute the triangles once per label and evaluate all gluings.
# ------------------------------------------------------------------------------------------
def all_gluings(s, y, what=('os0', 'ose', 'y0', 'hs', 'dece', 'hd'), De=None, ee=None):
    De = s.De if De is None else De
    ee = s.ee if ee is None else ee
    sb = s.sb
    D, hD = s.grids['D']; B, hB = s.grids['B']; S, hS = s.grids['S']; E, hE = s.grids['E']
    lD = logmu(D) - s.tD * D ** 2 + np.log(hD)
    out = {}
    need_B = any(w in what for w in ('os0', 'ose', 'y0', 'hs'))
    need_S = any(w in what for w in ('dece', 'hd'))
    if need_B:
        LIB = [s.logItri(a, y, D, B) for a in (0, 1)]
        lgcB = logg(s.Dchi, D[:, None], B[None, :])
        lWe0 = -sb * s.eps * B ** 2
        lWe = s.logWe(B, True, De, ee)
        for name, lw in (('os0', lWe0), ('ose', lWe)):
            if name in what:
                base = lD[:, None] + (logmu(B) + lw + np.log(hB))[None, :] - lgcB
                out[name] = {(a, b): lse((base + LIB[a] + np.conj(LIB[b])).ravel(), 0)
                             for a in (0, 1) for b in (0, 1)}
        lFB = lse(lD[:, None] + lgcB, 0)
        LSB = [lse(lD[:, None] + LIB[a], 0) for a in (0, 1)]
        for name, lw in (('y0', lWe0), ('hs', lWe)):
            if name in what:
                lS = logmu(B) + np.log(hB) - lFB + lw
                out[name] = {(a, b): lse(lS + LSB[a] + np.conj(LSB[b]), 0) for a in (0, 1) for b in (0, 1)}
    if need_S:
        LIS = [s.logItri(a, y, D, S) for a in (0, 1)]
        lgcS = logg(s.Dchi, D[:, None], S[None, :])
        lFS = lse(lD[:, None] + lgcS, 0)
        lge = logg(De, S[:, None], E[None, :])
        lP = lse((logmu(S) + np.log(hS) - sb * (2 * s.ts - 2 * ee) * S ** 2 + lFS)[:, None] + lge, 0)
        lE = logmu(E) + np.log(hE) - 2 * sb * ee * E ** 2 - lP
        lws = logmu(S) + np.log(hS) - sb * (s.ts + s.eps / 2 - 2 * ee - 1j * y) * S ** 2
        if 'dece' in what:
            LK = [lse((lws + lse(lD[:, None] + LIS[a], 0))[:, None] + lge, 0) for a in (0, 1)]
            out['dece'] = {(a, b): lse(lE + LK[a] + np.conj(LK[b]), 0) for a in (0, 1) for b in (0, 1)}
        if 'hd' in what:
            # hybrid: D shared between ket and bra, sigma_k, sigma_b separate.
            # Q_ab(sk,sb) = F(sk)F(sb)/N(sk,sb) * sum_D w(D) I_a(D,sk) conj I_b(D,sb)/sqrt(g(D,sk)g(D,sb))
            # N(sk,sb) = sum_D w(D) sqrt(g(D,sk) g(D,sb))
            hg = 0.5 * lgcS
            lN = lse(lD[:, None, None] + hg[:, :, None] + hg[:, None, :], 0)
            Xn = [LIS[a] - hg for a in (0, 1)]
            u = lws[:, None] + lge  # S x E  (ket factor; bra = conj)
            res = {}
            for a in (0, 1):
                for b in (0, 1):
                    M = lse(lD[:, None, None] + Xn[a][:, :, None] + np.conj(Xn[b])[:, None, :], 0)
                    Q = M + lFS[:, None] + lFS[None, :] - lN  # S x S
                    # rho = sum_E lE(E) sum_{sk,sb} u(sk,E) Q(sk,sb) conj u(sb,E)
                    # first contract sb:  R(sk,E) = lse_sb Q(sk,sb) + conj u(sb,E)
                    R = np.empty((len(S), len(E)), complex)
                    for i0 in range(0, len(S), 16):
                        R[i0:i0 + 16] = lse(Q[i0:i0 + 16, :, None] + np.conj(u)[None, :, :], 1)
                    res[(a, b)] = lse((u + R + lE[None, :]).ravel(), 0)
            out['hd'] = res
    return out
