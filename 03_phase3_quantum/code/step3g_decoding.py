"""
*** WARNING (16 Nov 2026): the identification of the shared replica energy with k_s below is WRONG -- it gives a
*** negative decoded variance (results/step3g_scan_FALSIFIED.txt). The shared region is the boundary cap after the
*** signal (see step3g_decoding.md, section 3). c1_micro and filter() are validated and stay in use.

Step 3g (open item 3) -- Hayden-Preskill / Petz decoding of the clock in the PSSY model.

Structural result used here (derivation in step3g_decoding.md):
  in the Petz replica geometry the stopwatch register rides on the clock's matter line, so the first-class planar
  geometries are rings of clock chords threading all replicas; all replicas share the energy of the region on the
  clock's side (k_s), each brane side (k_0) is separate.  Hence PSSY's planar machinery applies with
      y(s) -> y_c(k_s) = e^{-2 s tau_s k_s^2} * Int dmu(k0) e^{-t0 k0^2} g_m(k0, k_s)   (one code segment),
  and the decoded (normalised) clock observable is the physical one with the k_s-integrand reweighted by
      G(k_s) = w/(lambda_0 + w)          (PSSY 3.45; w = y_c/Z_1, s_k: k = Int_0^{s_k} rho-bar, lambda_0 = (1/k) Int_{s_k} rho-bar w),
  times the overall fidelity c_1 = Int rho-bar w G.   rho-bar = d_code e^{S0} dmu(k_s).
Also: exact microcanonical Petz coefficient from the Page (Marchenko-Pastur) spectrum, PSSY (2.38), (3.42).
"""
import numpy as np, sys
from scipy.special import loggamma
from scipy.integrate import quad
sys.path.insert(0, '03_phase3_quantum/code')
from step3d_backreaction import lmu, lg, saddle

def lse(L):
    o = np.max(np.real(L)); return o + np.log(np.sum(np.exp(L - o)))

# ------------------------------------------------------------------ microcanonical (PSSY 2.38, 3.42)
def c1_micro(k, N):
    """c1 = N^{-1} (Int sqrt(lam) D(lam))^2 with D the Page spectrum of rho_R (k x k, rank min(k,N))."""
    lm, lp = (k**-.5 - N**-.5)**2, (k**-.5 + N**-.5)**2
    f = lambda l: np.sqrt(l)*k*N/(2*np.pi*l)*np.sqrt(max((l - lm)*(lp - l), 0.0))
    I, _ = quad(f, lm, lp, limit=400, epsabs=0, epsrel=1e-11)
    return I**2/N

# ------------------------------------------------------------------ exact disk pieces, per k_s
class Decoder:
    def __init__(self, C, ts, m, S0, dcode=2, N=3000, kmax=None):
        self.C, self.ts, self.m, self.S0, self.d = C, ts, m, S0, dcode
        self.bM = np.pi/C; self.s = self.bM/(2*np.pi); s = self.s
        x1 = saddle(m/C, ts)[1]; kmax = kmax or (C*x1 + 10*np.sqrt(C*x1) + 10)
        self.k = np.linspace(1e-6, kmax, N); w = np.full(N, self.k[1] - self.k[0]); w[0] *= .5; w[-1] *= .5; self.wq = w
        self.t0 = self.bM - 2*s*ts
    def _inner(self, l, t, chunk=600):
        k, w = self.k, self.wq; off = np.empty(k.size); val = np.empty(k.size, complex); base = lmu(k) - k**2*t
        for i in range(0, k.size, chunk):
            E = base[None, :] + lg(l, k[i:i+chunk, None], k[None, :]); o = E.real.max(axis=1)
            off[i:i+chunk] = o; val[i:i+chunk] = (np.exp(E - o[:, None])*w[None, :]).sum(axis=1)
        return off, val
    def log_yc(self):
        """log of the code-segment weight y_c(k_s) (clock chord weight m, no signals)."""
        off, val = self._inner(self.m, self.t0)
        return off + np.log(val.real) - self.k**2*(2*self.s*self.ts)
    def integrand(self, dy, p):
        """log of the k_s-integrand of M_p (Step 3b), complex; M_p = Int dk_s exp(.)."""
        s = self.s; offA, A = self._inner(self.m - p/2, self.t0)
        off1, B1 = self._inner(p/2, s*(self.ts + 1j*dy)); B2 = np.conj(B1)
        return lmu(self.k) + offA + 2*off1 + np.log(A*B1*B2) + np.log(self.wq)
    def filter(self, logk):
        """PSSY canonical filter G(k_s), fidelity c1, and S_BH (clock-side ensemble) for radiation dimension e^{logk}."""
        k, wq = self.k, self.wq
        lrho = np.log(self.d) + self.S0 + lmu(k) + np.log(wq)          # rho-bar ds on the grid (log)
        ly = self.log_yc(); lZ1 = np.logaddexp.reduce(lrho + ly); lw = ly - lZ1   # w = y/Z1, Int rho-bar w = 1
        p = np.exp(lrho + lw)                                              # physical distribution of k_s (sums to 1)
        Sbh = -(p*(np.log(p) - lrho)).sum()                               # entropy of that distribution w.r.t. rho-bar
        cum = np.logaddexp.accumulate(lrho); i = np.searchsorted(cum, logk)
        i = min(max(i, 0), k.size - 1)
        lam0 = p[i:].sum()/np.exp(logk)
        G = 1/(1 + lam0*np.exp(-lw))
        return G, (p*G).sum(), Sbh, k[i]
    def integrand0(self):
        """log of the k_s-integrand of M_0 (the norm: clock chord only)."""
        off, val = self._inner(self.m, self.t0)
        return lmu(self.k) + off + np.log(val.real) - self.k**2*(2*self.s*self.ts) + np.log(self.wq)
    def decoded_moments(self, dys, ps, logk):
        G, c1, Sbh, sk = self.filter(logk); lG = np.log(G)
        L0 = self.integrand0()
        out = {}
        for p in ps:
            vals = []
            for dy in np.atleast_1d(dys):
                L = self.integrand(dy, p)
                phys = np.exp(lse(L) - lse(L0))*self.s**p
                dec = np.exp(lse(L + lG) - lse(L0 + lG))*self.s**p
                vals.append((phys, dec))
            out[p] = vals
        return out, c1, Sbh, sk
