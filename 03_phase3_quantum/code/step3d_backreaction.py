"""
Step 3d (open item 4) -- separating the clock's own classical backreaction from quantum-gravity effects
in the exact disk-level tick rate of Step 3b.

Exact object (Step 3b):  <R^p>_m = s^p M_p / M_0 ,  momentum integrals over k0 (preparation side, Euclidean length t0),
ks (signal side, t1 + t2 = 2 s tau_s) with the clock's bilocal O_{m-p/2} between them.

Classical limit at fixed mu = m/C (C -> infinity).  Write k0 = C x0, ks = C x1.  Stationarity of
    2 pi k0 + 2 pi ks - k0^2 t0 - ks^2 (t1+t2) + log Gamma(m +- i k0 +- i ks)
gives, with A = arctan((x0+x1)/mu), B = arctan((x0-x1)/mu),
    (1)  x0 (pi - tau_s) = pi - A - B          (preparation side)
    (2)  x1 tau_s        = pi - A + B          (signal side)
The light insertions (weight p/2) are then evaluated on this saddle (Fourier identity for |Gamma(l + i w)|^2), giving
    R_0(y; mu) = kappa1 sin(kappa1 tau_s) / [cosh(kappa1 (y - y_d)) - cos(kappa1 tau_s)] ,    kappa1 = x1 ,
i.e. the test-clock law at the temperature of the black hole *with the clock in it* (signal side), release depth
tau_s' = kappa1 tau_s.  The same equations give the exact junction condition of Step 2E.1:
    E1 - E0 = m eps_out + m^2 = m (eps_out + eps_in)/2 ,  eps_out = 2 k1 cot(tau_s'),  eps_in = 2 k0 cot(A + B).
First order in mu:
    kappa1 = 1 + (mu/pi) [1 + (pi - tau_s) cot tau_s] ,
    R_0/R_cl - 1 = (kappa1 - 1) [1 + tau_s cot tau_s - (dy sinh dy + tau_s sin tau_s)/(cosh dy - cos tau_s)].
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.special import loggamma

# ------------------------------------------------------------------ classical (saddle) part
def saddle(mu, ts):
    """(x0, x1) = (k0, ks)/C at the saddle, for mu = m/C."""
    if mu == 0: return 1.0, 1.0
    def eqs(v):
        x0, x1 = v
        A = np.arctan2(x0 + x1, mu); B = np.arctan2(x0 - x1, mu)
        return [x0*(np.pi - ts) - (np.pi - A - B), x1*ts - (np.pi - A + B)]
    k1 = 1 + mu/np.pi*(1 + (np.pi - ts)/np.tan(ts))
    x = fsolve(eqs, [k1 - mu/np.tan(ts), k1], xtol=1e-13)
    assert max(abs(np.array(eqs(x)))) < 1e-11
    return float(x[0]), float(x[1])

def R_cl(ts, dy, kap=1.0):
    return kap*np.sin(kap*ts)/(np.cosh(kap*dy) - np.cos(kap*ts))

def R0(mu, ts, dy):
    return R_cl(ts, dy, saddle(mu, ts)[1])

def kappa1_lin(ts):            # d kappa1 / d mu at mu = 0
    return (1 + (np.pi - ts)/np.tan(ts))/np.pi

def b_lin(ts, dy):             # d log R_0 / d mu at mu = 0  (the "slope/m" of Step 3b)
    return kappa1_lin(ts)*(1 + ts/np.tan(ts) - (dy*np.sinh(dy) + ts*np.sin(ts))/(np.cosh(dy) - np.cos(ts)))

def dy_horizon(tsp):           # ingoing label of the future-horizon crossing, kappa = 1 units; phi = 0 is at dy = 0
    return -2*np.arctanh(np.tan(tsp/2)**2)

def junction(mu, ts):
    """Check of the Step 2E.1 junction condition from the saddle (units of C^2; g = 2 in MTV units)."""
    x0, x1 = saddle(mu, ts)
    A = np.arctan2(x0 + x1, mu); B = np.arctan2(x0 - x1, mu)
    eo = 2*x1/np.tan(x1*ts); ei = 2*x0/np.tan(A + B)
    return dict(dE=x1**2 - x0**2, m_eps_out_plus_m2=mu*eo + mu**2, m_eps_avg=mu*(eo + ei)/2)

# ------------------------------------------------------------------ exact moments (Step 3b), chunked for large grids
def lmu(k):
    k = np.maximum(k, 1e-300)
    return np.log(2*k) + 2*np.pi*k + np.log1p(-np.exp(-4*np.pi*k)) - np.log(2)

def lg(l, a, b):
    return (2*loggamma(l + 1j*(a + b)).real + 2*loggamma(l + 1j*(a - b)).real
            - loggamma(2*l).real - np.log(2*np.pi**2))

class Exact:
    def __init__(self, C, N=4000, kmax=None, chunk=500):
        self.C = C; self.bM = np.pi/C; self.s = self.bM/(2*np.pi); self.N = N; self.kmax = kmax; self.chunk = chunk
    def grid(self, m, ts):
        if self.kmax: kmax = self.kmax
        else:
            x1 = saddle(m/self.C, ts)[1]
            kmax = self.C*x1 + 9*np.sqrt(self.C*x1) + 8          # saddle + 9 Gaussian widths (width^2 ~ C/ (2 pi) ...)
        k = np.linspace(0, kmax, self.N); w = np.full(self.N, k[1] - k[0]); w[0] *= .5; w[-1] *= .5
        return k, w
    def _inner(self, l, t, k, w):
        off = np.empty(len(k)); val = np.empty(len(k), complex); base = lmu(k) - k**2*t
        for i in range(0, len(k), self.chunk):
            ks = k[i:i + self.chunk]
            E = base[None, :] + lg(l, ks[:, None], k[None, :])
            o = E.real.max(axis=1); off[i:i + self.chunk] = o
            val[i:i + self.chunk] = (np.exp(E - o[:, None])*w[None, :]).sum(axis=1)
        return off, val
    def moments(self, ts, dys, m, ps):
        """{p: array over dys of <R^p>_m}.  The preparation-side integral is shared between labels."""
        s = self.s; t0 = self.bM - 2*s*ts; k, w = self.grid(m, ts); dys = np.atleast_1d(dys)
        E = lmu(k) - k**2*(2*s*ts); offA0, A0 = self._inner(m, t0, k, w); E0 = E + offA0; o0 = E0.max()
        M0 = (np.exp(E0 - o0)*A0*w).sum()
        out = {p: np.empty(len(dys), complex) for p in ps}
        for p in ps:
            offA, A = self._inner(m - p/2, t0, k, w)
            for j, dy in enumerate(dys):
                t1 = s*(ts + 1j*dy)
                off1, B1 = self._inner(p/2, t1, k, w)
                B2 = np.conj(B1)                                    # t2 = conj(t1), kernel real
                Ep = lmu(k) + offA + 2*off1; o = Ep.max()
                Mp = (np.exp(Ep - o)*A*B1*B2*w).sum()
                out[p][j] = Mp/M0*np.exp(o - o0)*s**p
        return out

# ------------------------------------------------------------------ physical parametrisation
def m_max(k1, tsp):
    """Largest clock mass that fits into a black hole of energy k1^2 when released at rest at depth tsp (E0 >= 0)."""
    return k1*np.tan(tsp/2)

def phys_to_bare(k1, tsp, m):
    """Black hole *with the clock in it*: S - S0 = 2 pi k1, release depth tsp (sin tsp = phi_h/phi_s), clock mass m.
    Returns the bare parameters (C, tau_s) of the exact formula and kappa1 = k1/C.  Uses the junction condition
    k0^2 = k1^2 - 2 m k1 cot(tsp) - m^2 and the saddle equation (1)."""
    E0 = k1**2 - 2*m*k1/np.tan(tsp) - m**2
    if E0 <= 0: raise ValueError(f"clock heavier than the black hole: m = {m} > m_max = {m_max(k1, tsp):.3f}")
    k0 = np.sqrt(E0); A = np.arctan2(k0 + k1, m); Bq = np.arctan2(k0 - k1, m)
    assert abs((np.pi - A + Bq) - tsp) < 1e-12
    t0 = (np.pi - A - Bq)/k0; bM = t0 + tsp/k1; C = np.pi/bM; x1 = k1/C
    return C, tsp/x1, x1

def energies(C, ts, m, N=4000):
    """Exact mean energies <k0^2> (preparation side) and <ks^2> (signal side: the black hole with the clock in it)
    in the state prepared by the clock insertion (weight M_0)."""
    X = Exact(C, N=N); s = X.s; t0 = X.bM - 2*s*ts; k, w = X.grid(m, ts)
    E = lmu(k)[:, None] + lmu(k)[None, :] - (k**2*t0)[:, None] - (k**2*2*s*ts)[None, :] + lg(m, k[:, None], k[None, :])
    W = np.exp(E - E.max())*w[:, None]*w[None, :]; Z = W.sum()
    return float((W*k[:, None]**2).sum()/Z), float((W*k[None, :]**2).sum()/Z)
