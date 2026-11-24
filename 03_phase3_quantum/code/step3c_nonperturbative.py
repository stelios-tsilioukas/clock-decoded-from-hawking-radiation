"""
Step 3c -- non-perturbative (random-matrix) corrections to the infalling clock's tick rate.

Disk level (step3b):  M_p = Int dmu(a) dmu(s) dmu(b) dmu(c) e^{-t0 E_a - t2 E_b - t1 E_c} g_{m-p/2}(a,s) g_{p/2}(s,b) g_{p/2}(s,c),
t1,2 = sc (tau_s +- i Dx),  Dx = x - y_d = real boundary label of the signal (ingoing OR outgoing).
Matrix-model completion (same prescription as Saad 1910.10311 and Franken-Mertens-Torres 2609.05164 for the two-point
function): the product of four densities becomes the four-density correlator of the matrix ensemble. Only the pair
(b,c) carries a real-time phase e^{-i T (E_c - E_b)}, T = sc*Dx. Its connected correlator gives
  * ramp   (genus-0 wormhole, exact JT double trumpet rho_{0,2}; in k-space  -(1/4 pi^2) f.p. 1/(k1-k2)^2 on the full line):
        corr_s(T) = (1/(8 pi^2)) Int dq |q| |Phi~_s(q)|^2,   Phi_s(k) = g_{p/2}(s,k) e^{-(sc tau_s - iT) k^2}
  * plateau (sine kernel):  corr_s -> corr_s - Int dE Phi0_s(E)^2 (|T|/2pi - rhobar(E))_+ ,  rhobar(E) = N sinh(2 pi sqrt E)
True density = N dmu/dE with N = e^{S0}/(8 pi^2) (SSS normalisation).  Relative correction = eps(T)/N^2 - (plateau cut).
All other pairs and the genus-one density corrections are O(1/N^2) with no secular growth for thermal T (see note).
"""
import numpy as np, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
from step3b_exact_rate import lmu, lg, R_cl

class NP:
    def __init__(self, C, ts, m, p=1, kmax=None, Nk=12001, Ns=241):
        self.C, self.ts, self.m, self.p = C, ts, m, p
        self.bM = np.pi/C; self.sc = self.bM/(2*np.pi)
        self.kmax = kmax or (6.5/np.sqrt(self.sc*ts) + 2.0)
        self.k = np.linspace(0, self.kmax, Nk); self.dk = self.k[1] - self.k[0]
        self.w = np.full(Nk, self.dk); self.w[0] *= 0.5; self.w[-1] *= 0.5
        self.s = np.linspace(0, self.kmax, Ns)[1:]; self.ds = self.s[1] - self.s[0]
        self.lg_sk = lg(p/2, self.s[:, None], self.k[None, :])                 # log g_{p/2}(s,k)
        self.lmu_k = lmu(self.k)
        # A(s) = Int dmu(a) e^{-t0 E_a} g_{m-p/2}(a,s)   (log-magnitude)
        t0 = self.bM - 2*self.sc*ts
        E = self.lmu_k[None, :] + lg(m - p/2, self.s[:, None], self.k[None, :]) - (self.k**2*t0)[None, :]
        off = E.max(axis=1); self.lA = off + np.log((np.exp(E - off[:, None])*self.w[None, :]).sum(axis=1))
        self.lmu_s = lmu(self.s)
    def disk(self, Dx):
        """M_p (disk) up to a Dx-independent constant: returns log-offset-free complex value normalised by exp(ref)."""
        t = self.sc*(self.ts + 1j*Dx)
        E = self.lmu_k[None, :] + self.lg_sk - (self.k**2*t)[None, :]
        off = E.real.max(axis=1); B = (np.exp(E - off[:, None])*self.w[None, :]).sum(axis=1)
        lw = self.lmu_s + self.lA + 2*off                                       # B1*B2 = |B|^2 (t2 = conj t1)
        return lw, np.abs(B)**2
    def worm(self, Dx):
        """Sum_s weight * corr_s(T): genus-0 wormhole in the pair (b,c)."""
        T = self.sc*Dx
        kk = np.concatenate([-self.k[:0:-1], self.k]); n = len(kk)
        nfft = 1 << int(np.ceil(np.log2(n))) + 2
        out = np.empty(len(self.s))
        ph = np.exp(-(self.sc*self.ts - 1j*T)*kk**2)
        q = 2*np.pi*np.fft.fftfreq(nfft, d=self.dk)
        for i in range(len(self.s)):
            g = np.exp(np.concatenate([self.lg_sk[i, :0:-1], self.lg_sk[i]]))
            F = np.fft.fft(g*ph, nfft)*self.dk
            out[i] = (np.abs(q)*np.abs(F)**2).sum()*(2*np.pi/(nfft*self.dk))/(8*np.pi**2)
        return self.lmu_s + self.lA, out
    def plateau_weight(self):
        """per s: arrays for Int dE Phi0(E)^2 (...) on the E = k^2 grid."""
        Phi2 = np.exp(2*self.lg_sk - 2*self.sc*self.ts*self.k[None, :]**2)      # Phi0(E)^2
        dE = 2*self.k*self.w
        return Phi2, dE
    def run(self, Dxs):
        res = []
        lwP, _ = self.worm(0.0)
        Phi2, dE = self.plateau_weight()
        ref = None
        for Dx in Dxs:
            lw, B2 = self.disk(Dx); lwW, cw = self.worm(Dx)
            ref = ref if ref is not None else lw.max()
            Md = (np.exp(lw - ref)*B2).sum()*self.ds
            Mw = (np.exp(lwW - ref)*cw).sum()*self.ds
            res.append(dict(Dx=float(Dx), disk=float(Md), worm=float(Mw)))
        # ramp slope and plateau ingredients (for asymptotics):  Sum_s weight * Int dE Phi0^2 ;  and E-resolved for the cut
        Es = self.k**2
        Wse = np.exp(lwP - ref)[:, None]*Phi2*self.ds                           # (s, E) weights
        self.ramp_slope = float((Wse*dE[None, :]).sum()/(2*np.pi))               # d(worm)/dT at large T
        self.E_profile = (Wse.sum(axis=0), dE, Es)                               # for the plateau cut
        return res
    def plateau_cut(self, T, N):
        prof, dE, Es = self.E_profile
        rb = N*np.sinh(2*np.pi*np.sqrt(Es))
        return float((prof*dE*np.clip(abs(T)/(2*np.pi) - rb, 0, None)).sum())
    def plateau_level(self, N):
        prof, dE, Es = self.E_profile
        return float((prof*dE*N*np.sinh(2*np.pi*np.sqrt(Es))).sum())

if __name__ == "__main__":
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    ts, m = np.pi/4, 2.0
    X = NP(C, ts, m)
    Dxs = np.concatenate([np.linspace(0, 1.2, 7), [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]])
    res = X.run(Dxs)
    d0 = res[0]['disk']
    print(f"C={C} (S-S0={2*np.pi*C:.2f}), tau_s=pi/4, m={m}; sc={X.sc:.4f}; kmax={X.kmax:.2f}")
    print("  Dx      <R>disk/Rcl     eps = N^2 * (wormhole/disk)    wormhole (abs, units of disk at Dx=0)")
    for r in res:
        r['R_over_cl'] = r['disk']/d0*R_cl(ts, 0.0)/R_cl(ts, r['Dx'])
        r['eps'] = r['worm']/r['disk']
        print(f" {r['Dx']:6.2f}   {r['R_over_cl']:10.4f}      {r['eps']:12.4e}        {r['worm']/d0:.4e}")
    print("ramp slope d(worm)/dT / disk(0) =", X.ramp_slope/d0, "  (check vs last two points:",
          (res[-1]['worm'] - res[-2]['worm'])/(X.sc*(res[-1]['Dx'] - res[-2]['Dx']))/d0, ")")
    out = dict(C=C, ts=ts, m=m, sc=X.sc, rows=res, ramp_slope_over_disk0=X.ramp_slope/d0, disk0=d0,
               plateau_level_over_disk0_per_N={str(N): X.plateau_level(N)/d0/N**2 for N in (1e2, 1e4, 1e6)})
    json.dump(out, open(f'03_phase3_quantum/results/step3c_np_C{C:g}.json', 'w'), indent=1)
