"""
Step 4g (21-22 Nov 2026): non-perturbative (random-matrix) corrections to the DECODED stopwatch reading.

Prescription (Step 3c, Saad 1910.10311, Franken-Mertens-Torres 2609.05164): random-matrix matter, fixed (non-crossing)
chord topology; the disk product of region densities becomes the spectral correlator of the JT matrix integral.
At relative order e^{-2 S0}: genus-one density corrections (one region at a time) and genus-0 wormholes (pairs).
This file computes the GENUS-ONE density corrections on all regions that are explicit in the Step 4b formulas:
  decoded  : hub s; P's ket and bra gaps; the reference gaps (Gam_ss, continued: weight -1); P's two brane caps D (inside S1, S2);
             the reference brane caps D (inside F_chi, weight -1)
  one-sided: the middle region B; the brane cap D; the emission-cap region inside B
Not included (stated in the note): the inner regions of the stop-signal triangles (caps A, C and the triangle interior)
and all pair wormholes.  Relative correction of region j:  delta_j = 4 pi^2 G1[F_j] / (disk),  F_j = integrand / dmu(k_j),
G1[F] = (1/192 pi) Int dq |q|^3 F~ + (pi/48) Int dq |q| F~   (finite parts of rho_{1,1}; checked on Gaussians against Z_{1,1}).
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step3h_replica import lse
from step4b_decorated import Decorated

class EvenFFT:
    """even extension of functions sampled on the uniform grid k_j = j dk (j = 0..N-1) and the genus-one functional."""
    def __init__(self, N, dk):
        self.N, self.dk = N, dk; self.NF = 2*N
        self.Q = 2*np.pi*np.fft.fftfreq(self.NF, d=dk); self.DQ = 2*np.pi/(self.NF*dk)
    def extend(self, F):                       # F[..., N] on k >= 0 -> F[..., 2N] on (j - N) dk
        out = np.zeros(F.shape[:-1] + (self.NF,), dtype=complex)
        out[..., self.N:] = F; out[..., 1:self.N] = F[..., :0:-1]
        return out
    def G1(self, F):                           # F on k >= 0, last axis
        Fe = self.extend(F); Ft = np.fft.fft(np.fft.ifftshift(Fe, axes=-1), axis=-1)*self.dk
        aq = np.abs(self.Q)
        return ((aq**3*Ft).sum(axis=-1)/(192*np.pi) + (aq*Ft).sum(axis=-1)*np.pi/48)*self.DQ

def at_zero(F):                                 # even function: F(0) from F(dk), F(2dk) (removes the 0/0 of integrand/mu)
    F = F.copy(); F[..., 0] = (4*F[..., 1] - F[..., 2])/3; return F

class GenusDecoded(Decorated):
    def prep(self):
        X = self.X; K = X.k; self.EF = EvenFFT(K.size, K[1] - K[0])
    def corrections(self, y):
        """disk rho_ab (deep post-Page) and one-sided F_ab, with the relative genus-one correction of every explicit region."""
        X = self.X; K = X.k; sb = X.s; eps, ee = self.eps, self.eps_e; EF = self.EF; four = 4*np.pi**2
        lmw = self.lmw; lw = self.lw; keep = self.keep
        Pk = {0: self.logI(self.D, y, +1), 1: self.logI(self.D + self.om, y, +1)}
        Pb = {0: self.logI(self.D, y, -1), 1: self.logI(self.D + self.om, y, -1)}
        corr = -np.pi*(K[:, None] + K[None, :])
        lgapP = lmw - K**2*sb*(self.ts + eps/2 - 2*ee)
        lB = lmw - K**2*sb*(eps - 2*ee) + self.lE
        # --- helper: relative correction factor for a sum S(col) = Sum_row exp(L[row, col]) w.r.t. the density of 'row'
        def rel_rows(L):
            o = lse(L, axis=0)                                                        # log S(col)
            Fr = np.exp(L - o[None, :] - lmw[:, None])                               # F(row; col) / S(col)  (row = k index)
            Fr[0, :] = 0.0; Fr = at_zero(Fr.T)                                       # (col, row)
            return four*EF.G1(Fr), o                                                 # delta S / S, per col
        def rel_cols(L):
            o = lse(L, axis=1)
            Fc = np.exp(L - o[:, None] - lmw[None, :]); Fc[:, 0] = 0.0; Fc = at_zero(Fc)
            return four*EF.G1(Fc), o
        # reference gaps and reference D
        dFx, lFx = rel_rows(self.lgx + self.colD)                                    # delta F_chi(g)/F_chi(g)
        Lss = self.lge + (lmw - K**2*sb*2*(self.ts - ee) + lFx)[None, :]             # Gam_ss(s) = Sum_g exp(Lss[s,g])
        dGss_gap, lGss = rel_cols(Lss)
        wss = np.exp(Lss - lGss[:, None])                                            # fraction of Gam_ss carried by each g
        dGss_D = (wss*dFx[None, :]).sum(axis=1)
        res = {}
        for a, b in ((0, 0), (1, 1), (0, 1)):
            M1, M2 = Pk[a] + corr, Pb[b] + corr
            # ---------------- decoded (deep)
            dS1, S1 = rel_rows(M1 + self.colD); dS2, S2 = rel_rows(M2 + self.colD)
            L1 = self.lge + (lgapP + 1j*K**2*sb*y + S1)[None, :]; L2 = self.lge + (lgapP - 1j*K**2*sb*y + S2)[None, :]
            dG1_gap, lG1 = rel_cols(L1); dG2_gap, lG2 = rel_cols(L2)
            w1 = np.exp(L1 - lG1[:, None]); w2 = np.exp(L2 - lG2[:, None])
            dG1_D = (w1*dS1[None, :]).sum(axis=1); dG2_D = (w2*dS2[None, :]).sum(axis=1)
            lY = lmw - K**2*sb*2*ee + lG1 + lG2 - lGss
            lY = np.where(keep, lY, -np.inf); lrho = lse(lY); wY = np.exp(lY - lrho)
            Fs = np.exp(lY - lrho - lmw); Fs[0] = 0.0; Fs = at_zero(Fs)
            dec = dict(hub=four*EF.G1(Fs),
                       gap_P=(wY*(dG1_gap + dG2_gap)).sum(), D_P=(wY*(dG1_D + dG2_D)).sum(),
                       gap_ref=-(wY*dGss_gap).sum(), D_ref=-(wY*dGss_D).sum())
            # ---------------- one-sided
            Lp = (M1 + M2 - self.lgx + self.colD + lB[None, :])
            Lp[0, :] = -np.inf; Lp[:, 0] = -np.inf
            lphys = lse(Lp)
            FD = np.exp(lse(Lp, axis=1) - lphys - lmw); FD[0] = 0.0; FD = at_zero(FD)
            FB = np.exp(lse(Lp, axis=0) - lphys - lmw); FB[0] = 0.0; FB = at_zero(FB)
            # emission cap index inside E(B): E(B) = Sum_k exp(lge[B,k] + lmw[k] - 2 ee sb k^2)
            dE, _ = rel_cols(self.lge + (lmw - K**2*sb*2*ee)[None, :])
            wB = np.exp(lse(Lp, axis=0) - lphys)
            one = dict(B=four*EF.G1(FB), D=four*EF.G1(FD), em=(wB*dE).sum())
            res[(a, b)] = dict(lrho=lrho, lphys=lphys, dec=dec, one=one)
        return res

def reading_shifts(res, om):
    """coefficient of e^{-2 S0} in the reading (arg V/omega) and in log|V|, per region and total."""
    out = {}
    for side in ('dec', 'one'):
        terms = res[(0, 1)][side].keys(); d = {}
        for t in terms:
            dV = res[(0, 1)][side][t] - 0.5*(res[(0, 0)][side][t] + res[(1, 1)][side][t])
            d[t] = dict(reading=float(np.imag(dV)/om), logabsV=float(np.real(dV)), rel_rho01=[float(np.real(res[(0, 1)][side][t])), float(np.imag(res[(0, 1)][side][t]))])
        tot = sum(res[(0, 1)][side][t] - 0.5*(res[(0, 0)][side][t] + res[(1, 1)][side][t]) for t in terms)
        d['total'] = dict(reading=float(np.imag(tot)/om), logabsV=float(np.real(tot)))
        V = np.exp(res[(0, 1)]['lrho' if side == 'dec' else 'lphys'] - 0.5*(res[(0, 0)]['lrho' if side == 'dec' else 'lphys'] + res[(1, 1)]['lrho' if side == 'dec' else 'lphys']))
        d['disk'] = dict(reading=float(np.angle(V)/om), absV=float(abs(V)))
        out[side] = d
    return out

def run(C, labels=('y_H-1', 'horizon', '0.9 inner'), ts=np.pi/4, D=2.0, h=1.0, eps=0.05, om=0.2, De=1.0, ee=0.0125, Nmul=1.0):
    x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    alllabs = {'y_H-1': yH - 1/x1, 'horizon': yH, '0.9 inner': -0.9*yH}
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/ee) + 10; N = int(Nmul*min(2800, max(1400, kmax/0.05)))
    S = GenusDecoded(C, ts, D, om, h, eps, kmax, N); S.setup_emission(De, ee); S.prep(); out = []
    for nm in labels:
        t0 = time.time(); r = reading_shifts(S.corrections(alllabs[nm]), om)
        out.append(dict(C=C, S_minus_S0=2*np.pi*C, label=nm, y=alllabs[nm], N=N, shifts=r))
        dd, oo = r['dec'], r['one']
        print(f"C={C:g} (S-S0={2*np.pi*C:.1f}) N={N} {nm:9s}: disk reading 1s {oo['disk']['reading']:+.5f} dec {dd['disk']['reading']:+.5f} | "
              f"coefficient of e^(-2 S0) in reading: one-sided {oo['total']['reading']:+.3e} [" + ", ".join(f"{t} {oo[t]['reading']:+.2e}" for t in ('B','D','em')) +
              f"]  decoded {dd['total']['reading']:+.3e} [" + ", ".join(f"{t} {dd[t]['reading']:+.2e}" for t in ('hub','gap_P','D_P','gap_ref','D_ref')) +
              f"] | log|V|: 1s {oo['total']['logabsV']:+.2e} dec {dd['total']['logabsV']:+.2e}  [{time.time()-t0:.0f}s]", flush=True)
    return out

if __name__ == "__main__":
    C = float(sys.argv[1]); Nmul = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    res = run(C, Nmul=Nmul)
    json.dump(res, open(f'04_phase4_radiating/results/step4g_genus_C{C:g}_N{Nmul:g}.json', 'w'), indent=1,
              default=lambda o: [float(np.real(o)), float(np.imag(o))] if np.iscomplexobj(o) else float(o))
