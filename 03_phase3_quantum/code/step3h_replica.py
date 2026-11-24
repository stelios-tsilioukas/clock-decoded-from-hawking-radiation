"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.

Step 3h (item 3, continued) -- Petz decoding of the clock register in the PSSY model with the cubic vertex.

Replica topology (derivation in step3h_replica_decoding.md):
  decoding sigma^+ (register 0 -> 1) puts the signal on the bra side of the physical boundary P; the register rides on
  the clock line, so clock lines thread the replicas (bra of one replica -> ket of the next, through the brane arc).
  Index flow: top half of the ring carries level 1 (psi_1), bottom half level 0 (psi_0).  Shared region = the region
  touching P's arc [u_+, y] and the middle arcs of all other replicas.  Region bookkeeping gives
      P's factor at shared energy s   =  T(s) = integrand of the one-sided amplitude <O_Delta(u_-) V(y) O_{Delta+omega}(u_+)>
                                             at fixed energy of the cap [u_+, y]           (fan chords (h+w)/2, (h-w)/2, Delta+(w-h)/2)
      code replica of level a           =  y_a(s) = integrand of <O_a(u_-) O_a(u_+)> at fixed energy of the arc [u_+, u_-].
  First-class planar sum (PSSY 3.37-3.40, generalised to level-dependent weights), PSSY canonical approximation of the
  lambda integrals (3.44-3.45):
      C_dec = [ Sum_s rho w_P w_D (lam0 + w_1)^{-1/2} (lam0 + w_0)^{-1/2} ] / [ Sum_s rho w_0^2/(lam0 + w_0) ],
      w_a = P_a/rho, P_a = y_a/Z_a,  w_P = T/(Z_0 rho),  w_D = P_0 sqrt(Z_0/Z_1)/rho.
  Petz value / deep post-Page (lam0 -> 0):  C_dec = Sum_s T sqrt(P_0/P_1) / sqrt(Z_0 Z_1)
  Physical one-sided (Step 3e) value:       C_phys = Sum_s T / sqrt(Z_0 Z_1).
"""
import numpy as np, sys
sys.path.insert(0, '03_phase3_quantum/code')
from step3d_backreaction import lmu, lg, saddle

def lse(L, axis=None):
    # log-sum-exp; an all -inf slice returns -inf (not nan): audit M3, 21 Nov 2026
    o = np.max(np.real(L), axis=axis, keepdims=True)
    o = np.where(np.isfinite(o), o, 0.0)
    r = o + np.log(np.sum(np.exp(L - o), axis=axis, keepdims=True))
    return np.squeeze(r, axis=axis) if axis is not None else r.item()

class Replica:
    def __init__(self, C, ts, m, S0=8.0, dcode=2, N=2400, kmax=None):
        self.C, self.ts, self.m, self.S0, self.d = C, ts, m, S0, dcode
        self.bM = np.pi/C; self.s = self.bM/(2*np.pi)
        x1 = saddle(m/C, ts)[1]; kmax = kmax or (C*x1 + 10*np.sqrt(C*x1) + 12)
        self.k = np.linspace(1e-6, kmax, N); w = np.full(N, self.k[1] - self.k[0]); w[0] *= .5; w[-1] *= .5
        self.lwq = np.log(w); self.lmu = lmu(self.k); self.t0 = self.bM - 2*self.s*ts
        self.lrho = np.log(dcode) + S0 + self.lmu + self.lwq                    # rho-bar per grid point
    def _inner(self, l, t, chunk=600):
        """log Int dmu(k) e^{-t k^2} g_l(s,k), for every grid s."""
        k = self.k; base = self.lmu + self.lwq - k**2*t; out = np.empty(k.size, complex)
        for i in range(0, k.size, chunk):
            out[i:i+chunk] = lse(base[None, :] + lg(l, k[i:i+chunk, None], k[None, :]), axis=1)
        return out
    def log_y(self, ma):
        """code-state weight of a clock of dimension ma, per grid point of the energy of the arc [u_+, u_-]."""
        return self.lmu + self.lwq - self.k**2*(2*self.s*self.ts) + self._inner(ma, self.t0).real
    def log_T(self, dy, wa, wb, wc, chunk=600):
        """fan integrand at fixed energy k1 of the cap [u_+, y] (chord (+,y)=wa, (y,-)=wb, (+,-)=wc), per grid point."""
        s = self.s; k = self.k
        v = self.lmu + self.lwq + self._inner(wc, self.t0) + self._inner(wb, s*(self.ts - 1j*dy))   # over k_s
        out = np.empty(k.size, complex)
        for i in range(0, k.size, chunk):
            out[i:i+chunk] = lse(lg(wa, k[i:i+chunk, None], k[None, :]) + v[None, :], axis=1)
        return out + self.lmu + self.lwq - k**2*(s*(self.ts + 1j*dy))
    # ---------------------------------------------------------------- decoding
    def lam0(self, lP0, lP1, logk):
        """PSSY (2.39), (2.43) for the code-averaged ensemble."""
        cum = np.logaddexp.accumulate(self.lrho); i = int(np.searchsorted(cum, logk)); i = min(max(i, 0), self.k.size - 1)
        lPbar = np.logaddexp(lP0, lP1) - np.log(2)
        return (np.exp(lse(lPbar[i:]) - logk) if i < self.k.size - 1 else 0.0), self.k[i]
    def decode(self, lT, ly0, ly1, logk):
        """returns (C_dec, C_phys, C_deep, c1, s_k) for the coherence with P-factor T and code levels y0, y1."""
        lZ0, lZ1 = lse(ly0), lse(ly1); lP0, lP1 = ly0 - lZ0, ly1 - lZ1
        lw0, lw1 = lP0 - self.lrho, lP1 - self.lrho
        lam0, sk = self.lam0(lP0, lP1, logk) if np.isfinite(logk) else (0.0, np.inf)
        ll0 = np.log(lam0) if lam0 > 0 else -np.inf
        a0, a1 = np.logaddexp(ll0, lw0), np.logaddexp(ll0, lw1)
        num = lse(lT - lZ0 + lP0 + 0.5*(lZ0 - lZ1) - self.lrho - 0.5*a1 - 0.5*a0)
        den = lse(lP0 + lw0 - a0)
        Cdec = np.exp(num - den); Cphys = np.exp(lse(lT) - 0.5*(lZ0 + lZ1))
        Cdeep = np.exp(lse(lT + 0.5*(lP0 - lP1)) - 0.5*(lZ0 + lZ1))
        return Cdec, Cphys, Cdeep, np.exp(den), sk
