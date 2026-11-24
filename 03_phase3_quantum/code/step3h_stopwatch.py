"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.

Step 3h: decoding a frozen stopwatch.  Running levels psi_0 (Delta), psi_1 (Delta + omega) prepared in superposition at u_+;
stop signal V (dim h) at y converts psi_a -> chi_a, with chi_0, chi_1 degenerate (dim Dchi = Delta + omega/2).
Register = chi label (gravitationally inert).  Coherence rho_01 and populations rho_aa:
  physical (one-sided):  F_ab = Sum_{s,D} I_1^(a)(D,s) I_2^(b)(D,s)/g_chi(D,s)     (Step 3f gluing, D shared)
  decoded (Petz, replica ring of chi lines): Y_ab(s) = mu e^{-t_B s^2} S_1^(a)(s) S_2^(b)(s)/F_chi(s),  S = Int_D I,
      rho_ab = Sum_s Y_ab(s) G_chi(s)   (all code levels have the same weight -> one filter G_chi)
Visibility V = rho_01/sqrt(rho_00 rho_11) (<= 1 by positivity), reading = arg V/omega.
"""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3f_ramsey as F, step3d_backreaction as B
from step3h_replica import lse

def tri_weights(Dext, h, Dx):
    """triangle (ext, y, X): chord adjacent to the cap, to the central region B, to D."""
    return (Dext + h - Dx)/2, (h + Dx - Dext)/2, (Dext + Dx - h)/2

class Stopwatch:
    def __init__(self, C, ts, D, om, h, eps, kmax, N, S0=8.0, dcode=2):
        self.X = X = F.ExactRamsey(C, kmax, N); self.ts, self.D, self.om, self.h, self.eps = ts, D, om, h, eps
        s = X.s; K = X.k; self.lw = np.log(X.w); self.lmu = X.lmuh + 2*np.pi*K; self.tD = X.bM - 2*s*ts
        self.Dx = D + om/2; self.MX = X.MX
        self.lgx = F.lg(self.Dx, K[:, None], K[None, :])
        self.colD = (self.lmu + self.lw - K**2*self.tD)[:, None]
        self.lFx = lse(self.lgx + self.colD, axis=0)                               # log F_chi(s)
        ly = self.lmu + self.lw - K**2*(2*s*ts) + X.log_inner(self.Dx, self.tD).real   # code weight of a chi clock
        keep = K > 0; self.keep = keep
        self.lrho = (np.log(dcode) + S0 + self.lmu + self.lw)[keep]; lyk = ly[keep]
        self.lP = lyk - lse(lyk); self.lwc = self.lP - self.lrho
        self.Sbh = -(np.exp(self.lP)*(self.lP - self.lrho)).sum()
    def logI(self, Dext, y, side):
        """log I(D, s) + pi(D + s) for the ket (side=+1, cap A) or bra (side=-1, cap C) triangle."""
        X = self.X; s = X.s; a, b, c = tri_weights(Dext, self.h, self.Dx)
        t = s*(self.ts - self.eps/2 + 1j*y) if side > 0 else s*(self.ts - self.eps/2 - 1j*y)
        return X.logLmR(X.lghat(c), X.lghat(b), X.log_inner(a, t))
    def elements(self, y):
        X = self.X; K = X.k; s = X.s; tB = s*self.eps
        Pk = {0: self.logI(self.D, y, +1), 1: self.logI(self.D + self.om, y, +1)}
        Pb = {0: self.logI(self.D, y, -1), 1: self.logI(self.D + self.om, y, -1)}
        corr = -np.pi*(K[:, None] + K[None, :])
        rowB = (self.lmu + self.lw - K**2*tB)[None, :]
        phys, Y = {}, {}
        for a, b in ((0, 0), (1, 1), (0, 1)):
            M1, M2 = Pk[a] + corr, Pb[b] + corr                                   # log I (D, s)
            phys[(a, b)] = lse((M1 + M2 - self.lgx + self.colD + rowB)[1:, 1:])
            S1 = lse(M1 + self.colD, axis=0); S2 = lse(M2 + self.colD, axis=0)
            Y[(a, b)] = (self.lmu + self.lw - K**2*tB + S1 + S2 - self.lFx)[self.keep]
        return phys, Y
    def filter(self, x):
        if not np.isfinite(x): return np.zeros(self.lrho.size)
        logk = self.Sbh + x; cum = np.logaddexp.accumulate(self.lrho); i = min(int(np.searchsorted(cum, logk)), self.lrho.size - 1)
        lam0 = np.exp(lse(self.lP[i:]) - logk) if i < self.lrho.size - 1 else 0.0
        return self.lwc - np.logaddexp(np.log(lam0) if lam0 > 0 else -np.inf, self.lwc)   # log G
    def visibility(self, y, xs):
        phys, Y = self.elements(y)
        Vp = np.exp(phys[(0, 1)] - 0.5*(phys[(0, 0)] + phys[(1, 1)]))
        out = []
        for x in xs:
            lG = self.filter(x); r = {k: lse(v + lG) for k, v in Y.items()}
            Vd = np.exp(r[(0, 1)] - 0.5*(r[(0, 0)] + r[(1, 1)]))
            out.append(Vd)
        return Vp, out

if __name__ == "__main__":
    ts = np.pi/4; D = 2.0; h = 1.0; eps = 0.05; res = []
    for C in (10.0, 20.0, 40.0):
        x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
        labs = {'y_H-1': yH - 1/x1, 'horizon': yH, 'phi=0': 0.0, '0.9 inner': -0.9*yH}
        kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2800, max(1400, kmax/0.04)))
        for om in (0.2,):
            S = Stopwatch(C, ts, D, om, h, eps, kmax, N)
            line = {}
            for nm, y in labs.items():
                Vp, (Vd,) = S.visibility(y, [np.inf])
                line[nm] = dict(Vphys=[abs(Vp), np.angle(Vp)/om], Vdec=[abs(Vd), np.angle(Vd)/om])
            res.append(dict(C=C, omega=om, N=N, labels=labs, values=line))
            print(f"C={C:4.0f} omega={om} N={N}: " + " | ".join(f"{nm}: |V| phys {v['Vphys'][0]:.5f} dec {v['Vdec'][0]:.5f}; reading phys {v['Vphys'][1]:+.5f} dec {v['Vdec'][1]:+.5f} (C x diff {C*(v['Vdec'][1]-v['Vphys'][1]):+.4f})" for nm, v in line.items()), flush=True)
    json.dump(res, open('03_phase3_quantum/results/step3h_stopwatch_deep.json', 'w'), indent=1, default=float)
