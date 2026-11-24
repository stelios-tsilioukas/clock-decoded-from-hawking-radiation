"""
Step 3b (first exact result) -- the tick rate of the infalling clock in the exact Schwarzian theory.

Classical identity (beta = 2 pi; Euclidean labels u_+ = -tau_s + i y_d, u_y = i y, u_- = +tau_s + i y_d):
    R(y) := d tau / d y = O_{1/2}(y,+) O_{1/2}(-,y) / O_{1/2}(-,+) = sin(tau_s) / (cosh(y - y_d) - cos(tau_s)),
with O_l the Schwarzian bilocal.  Three bilocals forming a triangle do not cross, so the moments
    <R^p>_m = M_p / M_0,   M_p = < O_{p/2}(y,+) O_{p/2}(-,y) O_{m-p/2}(-,+) >,   M_0 = < O_m(-,+) >
are exact momentum integrals (Mertens-Turiaci-Verlinde 1705.08408, rules (1.16)-(1.19)):
    M_p = Int dmu(k0) dmu(k1) dmu(k2) dmu(ks) e^{-k0^2 t0 - k1^2 t1 - k2^2 t2}
              g_{m-p/2}(k0,ks) g_{p/2}(ks,k1) g_{p/2}(ks,k2),      g_l(a,b) = Gamma(l +- ia +- ib)/Gamma(2l)/(2 pi^2),
    dmu(k) = 2k sinh(2 pi k) dk,  E = k^2 (MTV units, C_MTV = 1/2),
    beta_M = pi / C  (C = MSY coupling at beta = 2 pi;  S - S0 = 2 pi C = 2 pi^2/beta_M),
    t1 = s (tau_s + i(y - y_d)),  t2 = s (tau_s - i(y - y_d)),  t0 = beta_M - 2 s tau_s,  s = beta_M/(2 pi).
m = mass (conformal weight) of the clock particle; needs m > p/2.
"""
import numpy as np, json, sys
from scipy.special import loggamma

def lmu(k):                       # log of 2k sinh(2 pi k)
    k = np.maximum(k, 1e-300)
    return np.log(2*k) + 2*np.pi*k + np.log1p(-np.exp(-4*np.pi*k)) - np.log(2)

def lg(l, a, b):                  # log of Gamma(l +- ia +- ib)/Gamma(2l)/(2 pi^2), real for l > 0
    return (2*loggamma(l + 1j*(a + b)).real + 2*loggamma(l + 1j*(a - b)).real
            - loggamma(2*l).real - np.log(2*np.pi**2))

class Exact:
    def __init__(self, C, kmax=None, N=2400):
        self.C = C; self.bM = np.pi/C; self.s = self.bM/(2*np.pi)
        self.N = N; self.kmax = kmax
    def grid(self, tmin):
        kmax = self.kmax or (1.6*2*np.pi/self.bM + 7/np.sqrt(tmin) + 6)
        k = np.linspace(0, kmax, self.N); w = np.full(self.N, k[1] - k[0]); w[0] *= 0.5; w[-1] *= 0.5
        return k, w
    def _inner(self, l, t, k, w):
        """log-magnitude offset and complex value of  I(ks) = Int dmu(k) e^{-k^2 t} g_l(ks,k)  for every ks on the grid."""
        E = lmu(k)[None, :] + lg(l, k[:, None], k[None, :]) - (k**2*t)[None, :]
        off = E.real.max(axis=1)
        val = (np.exp(E - off[:, None])*w[None, :]).sum(axis=1)
        return off, val
    def two_point(self, l, tau_u):
        """<O_l> at Euclidean separation tau_u (in beta = 2 pi units), normalised by Z."""
        t = self.s*tau_u; k, w = self.grid(min(abs(t), self.bM - abs(t)) if np.isreal(t) else abs(t.real))
        off, val = self._inner(l, t, k, w)
        E0 = lmu(k) - k**2*(self.bM - t) + off; o = E0.real.max()
        num = (np.exp(E0 - o)*val*w).sum()
        Ez = lmu(k) - k**2*self.bM; oz = Ez.max(); Z = (np.exp(Ez - oz)*w).sum()
        return num/Z*np.exp(o - oz)
    def moments(self, ts, dy, m, ps):
        """<R^p>_m for the listed p (R in beta = 2 pi units: d tau / d y)."""
        s = self.s; t1 = s*(ts + 1j*dy); t2 = s*(ts - 1j*dy); t0 = self.bM - 2*s*ts
        k, w = self.grid(min(t1.real, t0))
        def M(p):
            offA, A = self._inner(m - p/2, t0, k, w)
            if p == 0:
                # g_0 -> delta function: M_0 = <O_m(-,+)> ; ks integral collapses onto k1 = k2 = ks
                E = lmu(k) - k**2*(t1 + t2) + offA; o = E.real.max()
                return o, (np.exp(E - o)*A*w).sum()
            off1, B1 = self._inner(p/2, t1, k, w); off2, B2 = self._inner(p/2, t2, k, w)
            E = lmu(k) + offA + off1 + off2; o = E.real.max()
            return o, (np.exp(E - o)*A*B1*B2*w).sum()
        o0, M0 = M(0); out = {}
        for p in ps:
            o, Mp = M(p); out[p] = Mp/M0*np.exp(o - o0)*s**p          # s^p: MTV time units -> beta = 2 pi units
        return out

def R_cl(ts, dy): return np.sin(ts)/(np.cosh(dy) - np.cos(ts))

if __name__ == "__main__":
    # ---- normalisation check: exact two-point function vs classical (1/(2 sin(u/2)))^{2l} at large C
    for C in (10.0, 40.0):
        X = Exact(C)
        for l, u in ((0.5, 1.0), (1.0, 2.0), (2.5, 0.7)):
            ex = X.two_point(l, u); cl = (1/(2*np.sin(u/2)))**(2*l)/X.s**(2*l)
            print(f"C={C} l={l} u={u}: exact/classical = {ex.real/cl:.5f}   (imag {ex.imag:.1e})")
