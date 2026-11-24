"""
Independent re-derivation (agent stopwatch-split): one-sided frozen-stopwatch reading.

Exact disk formula (derived in report.md):

 F_ab = int dmu(kB) dmu(kD) exp(-s[eps kB^2 + (2pi-2tau) kD^2]) K_a(kB,kD;lA) K_b(kB,kD;lC) / g_chi(kB,kD)
 K_a(kB,kD;l) = int dmu(ki) g_{w2}(kB,ki) g_{w3}(kD,ki) H_a(ki;l)
 H_a(ki;l)    = int dmu(kA) exp(-s l kA^2) g_{w1}(kA,ki)
 lA = tau + i y - eps/2,  lC = tau - i y - eps/2 = conj(lA),  s = 1/(2C)
 w1 = (D_a + h - D_chi)/2, w2 = (h + D_chi - D_a)/2, w3 = (D_chi + D_a - h)/2

Everything even in each momentum -> midpoint rule on [0,kmax] is spectrally accurate.
"""
import numpy as np
from scipy.special import loggamma
from scipy.optimize import brentq, fsolve

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- classical
def R_cl(tau, y, eps):
    """fixed-geometry reading = arg[sin((tau-eps/2+iy)/2) sin((tau+eps/2+iy)/2)] (omega-independent)"""
    z1 = np.sin((tau - eps / 2 + 1j * y) / 2)
    z2 = np.sin((tau + eps / 2 + 1j * y) / 2)
    return np.angle(z1) + np.angle(z2)


def yH_of(tau_phys):
    """horizon label (in units of the actual temperature)"""
    return -2 * np.arctanh(np.tan(tau_phys / 2) ** 2)


# ---------------------------------------------------------------- physical-variable map
def phys_to_bare(k1, taup, m):
    k0sq = k1 ** 2 - 2 * m * k1 / np.tan(taup) - m ** 2
    k0 = np.sqrt(k0sq)
    A = np.arctan2(k0 + k1, m)
    B = np.arctan2(k0 - k1, m)
    t0 = (np.pi - A - B) / k0
    betaM = t0 + taup / k1
    C = np.pi / betaM
    kap = k1 / C
    tau = taup / kap
    return dict(k0=k0, A=A, B=B, check_taup=np.pi - A + B, t0=t0, betaM=betaM, C=C, kappa1=kap, tau=tau)


def bare_to_phys(C, tau, m):
    """solve for (k1, tau') given bare (C, tau) and clock mass m (1d root find in k1, tau' = k1 tau / C)"""
    f = lambda k1: phys_to_bare(k1, k1 * tau / C, m)['C'] - C
    lo = C
    hi = C + 0.5 * m
    while f(hi) < 0:
        hi += 0.5 * m
    # need k0^2>0 on bracket
    k1 = brentq(f, lo + 1e-12, hi, xtol=1e-15, rtol=1e-15)
    d = phys_to_bare(k1, k1 * tau / C, m)
    d['k1'], d['taup'] = k1, k1 * tau / C
    return d


# ---------------------------------------------------------------- exact numerics
def _log_absgamma2_table(w, dk, nmax):
    """L[n] = log|Gamma(w + i n dk)|^2  for n = -nmax..nmax (index n+nmax)"""
    n = np.arange(-nmax, nmax + 1)
    return 2 * np.real(loggamma(w + 1j * n * dk))


class Grid:
    def __init__(self, dk, N):
        self.dk, self.N = dk, N
        self.j = np.arange(N)
        self.k = (self.j + 0.5) * dk
        k = self.k
        self.logmu = np.log(2 * k) + TWO_PI * k + np.log1p(-np.exp(-2 * TWO_PI * k)) - np.log(2.0) + np.log(dk)
        # log(2k sinh 2pi k dk) = log(2k) + 2pi k + log(1-e^{-4pi k}) - log 2 + log dk
        self._tabs = {}

    def logg(self, w, n1=None, n2=None):
        """log g_w(k_a,k_b) matrix for a<n1, b<n2 (shared midpoint grid)"""
        n1 = n1 or self.N
        n2 = n2 or self.N
        key = round(w, 14)
        if key not in self._tabs:
            self._tabs[key] = _log_absgamma2_table(w, self.dk, 2 * self.N + 2)
        T = self._tabs[key]
        off = 2 * self.N + 2
        ja = self.j[:n1, None]
        jb = self.j[None, :n2]
        # k_a + k_b = (ja+jb+1) dk ; k_a - k_b = (ja-jb) dk
        L = T[ja + jb + 1 + off] + T[ja - jb + off]
        return L - np.log(2 * np.pi ** 2) - np.real(loggamma(2 * w))


def _cplx_logsumexp(L, axis):
    m = np.max(L.real, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    S = np.sum(np.exp(L - m), axis=axis)
    return np.log(S) + np.squeeze(m, axis=axis)


def logK(grid, s, ell, w1, w2, w3, NB):
    """returns log K(kB,kD) (complex, shape NB x N) and diagnostics"""
    k = grid.k
    N = grid.N
    # H(i) = sum_A mu_A exp(-s l A^2) g_w1(A,i)
    L1 = grid.logg(w1)                       # [A, i]
    LA = grid.logmu - s * ell * k ** 2        # [A]
    logH = _cplx_logsumexp(L1 + LA[:, None], axis=0)   # [i]
    c = grid.logmu + logH                     # [i]
    L2 = grid.logg(w2, NB, N)                 # [B, i]
    L3 = grid.logg(w3, N, N)                  # [D, i]
    half = 0.5 * c.real
    X2 = L2 + half[None, :]
    X3 = L3 + half[None, :]
    rB = X2.max(axis=1)
    qD = X3.max(axis=1)
    M2 = np.exp(X2 - rB[:, None])
    M3 = np.exp(X3 - qD[:, None]) * np.exp(1j * c.imag)[None, :]
    S = M2 @ M3.T.real + 1j * (M2 @ M3.T.imag)     # [B, D]
    with np.errstate(divide='ignore'):
        LK = np.log(S) + rB[:, None] + qD[None, :]
    return LK


def exact_F(C, tau, y, eps, Delta, omega, h, dk=0.1, kmax=None, kmaxB=None, return_all=False):
    s = 1.0 / (2 * C)
    if kmax is None:
        kmax = C + 9 * np.sqrt(C) + 3 * Delta + 10
    if kmaxB is None:
        kmaxB = kmax + 80
    N = int(np.ceil(kmax / dk))
    NB = int(np.ceil(kmaxB / dk))
    gridm = Grid(dk, NB)
    Dch = Delta + omega / 2
    Ds = [Delta, Delta + omega]
    lA = tau + 1j * y - eps / 2
    LK = []
    for a in (0, 1):
        Da = Ds[a]
        w1 = (Da + h - Dch) / 2
        w2 = (h + Dch - Da) / 2
        w3 = (Dch + Da - h) / 2
        # compute with i,A,D restricted to first N points, B up to NB
        LK.append(_logK_restricted(gridm, N, NB, s, lA, w1, w2, w3))
    kB = gridm.k[:NB]
    kD = gridm.k[:N]
    LW = (gridm.logmu[:NB, None] + gridm.logmu[None, :N]
          - s * (eps * kB[:, None] ** 2 + (TWO_PI - 2 * tau) * kD[None, :] ** 2)
          - gridm.logg(Dch, NB, N))
    out = {}
    for (a, b) in [(0, 0), (1, 1), (0, 1)]:
        T = LW + LK[a] + np.conj(LK[b])
        m = np.nanmax(T.real[np.isfinite(T.real)])
        Z = np.exp(T - m)
        Z[~np.isfinite(T.real)] = 0
        out[(a, b)] = (np.sum(Z), m)
        if return_all:
            out[('Z', a, b)] = Z
    F00, m00 = out[(0, 0)]
    F11, m11 = out[(1, 1)]
    F01, m01 = out[(0, 1)]
    Vis = F01 / np.sqrt(F00 * F11) * np.exp(m01 - 0.5 * (m00 + m11))
    res = dict(V=Vis, reading=np.angle(Vis) / omega, absV=abs(Vis), N=N, NB=NB,
               imF00=np.angle(F00), imF11=np.angle(F11))
    if return_all:
        res['out'] = out
        res['grid'] = gridm
    return res


def _logK_restricted(grid, N, NB, s, ell, w1, w2, w3):
    k = grid.k
    L1 = grid.logg(w1, N, N)                  # [A, i]
    LA = grid.logmu[:N] - s * ell * k[:N] ** 2
    logH = _cplx_logsumexp(L1 + LA[:, None], axis=0)
    c = grid.logmu[:N] + logH
    L2 = grid.logg(w2, NB, N)                 # [B, i]
    L3 = grid.logg(w3, N, N)                  # [D, i]
    half = 0.5 * c.real
    X2 = L2 + half[None, :]
    X3 = L3 + half[None, :]
    rB = X2.max(axis=1)
    qD = X3.max(axis=1)
    M2 = np.exp(X2 - rB[:, None])
    ph = np.exp(1j * c.imag)
    M3r = np.exp(X3 - qD[:, None]) * ph.real[None, :]
    M3i = np.exp(X3 - qD[:, None]) * ph.imag[None, :]
    S = M2 @ M3r.T + 1j * (M2 @ M3i.T)
    with np.errstate(divide='ignore'):
        LK = np.log(S) + rB[:, None] + qD[None, :]
    return LK
