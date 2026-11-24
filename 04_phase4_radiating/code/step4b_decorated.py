"""
Step 4b: the decoded frozen stopwatch of the INFALLING clock (tau_s = pi/4) with finite-Delta_e late emission.

Late-time, large-k2 configuration (Step 4a): clock (chi) lines and emission lines both thread the replicas; emission
chords are nested between the chi chords and the centre.  Regions:
  hub  s : touches every replica's emission arc [e_ket, e_bra] (Euclidean 2 eps_e), shared by all replicas;
  gap  g : between consecutive replicas j|j+1; bounded by the emission chord e_bra(j)->e_ket(j+1) (bilocal g_De(s,g))
           and by the chi line (and, next to P, the stop-signal lines and vertex);
  brane caps D (local), P's caps A, C (local).
Per-replica factors (all functions of the hub energy s; Lorentzian times cancel inside each gap except P's +-iy):
  sigma/D (reference chi code):  w(s)  ~ mu(s) e^{-2 eps_e sb s^2} Gam_ss(s),
     Gam_ss(s)   = Sum_g mu w_g e^{-sb g^2 2(ts - eps_e)} g_De(s,g) F_chi(g)
  P (decoded register a,b):  Y_ab(s) = mu(s) e^{-2 eps_e sb s^2} Gam_sP^(a)(s) Gam_Ps^(b)(s) / Gam_ss(s),
     Gam_sP^(a)(s) = Sum_g mu w_g e^{-sb g^2 (ts + eps/2 - 2 eps_e - i y)} g_De(s,g) S_1^(a)(g)
     Gam_Ps^(b)(s) = Sum_g mu w_g e^{-sb g^2 (ts + eps/2 - 2 eps_e + i y)} g_De(s,g) S_2^(b)(g)
  rho_ab = Sum_s Y_ab(s) G(s),  G = w J^2 (exact resolvent of the hub ensemble);  G -> 1 deep post-Page.
  Check: g_De -> delta (De -> 0) gives Y -> mu e^{-eps sb s^2} S1 S2 / F_chi = Step 3h exactly.
Physical (one-sided) register after the channel: Step 3f gluing with the emission cap inside region B:
  F_ab = Sum_{D,B} I1 I2 / g_chi  mu_B e^{-sb B^2 (eps - 2 eps_e)} E(B),  E(B) = Sum_k mu w e^{-2 eps_e sb k^2} g_De(B,k).
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_stopwatch as SW, step3f_ramsey as F, step3d_backreaction as B
from step3h_replica import lse
from step3i_resolvent import ExactResolvent

OUT = '04_phase4_radiating/results/'

class Decorated(SW.Stopwatch):
    _memo = None
    def logI(self, Dext, y, side):
        if self._memo is None: self._memo = {}
        key = (round(Dext, 12), round(y, 12), side)
        if key not in self._memo: self._memo[key] = SW.Stopwatch.logI(self, Dext, y, side)
        return self._memo[key]
    def setup_emission(self, De, eps_e):
        X = self.X; K = X.k; sb = X.s; self.De, self.eps_e = De, eps_e
        self.lge = F.lg(De, K[:, None], K[None, :])                       # log g_De(s, g): rows hub s, cols gap g
        self.lmw = self.lmu + self.lw
        self.lGss = lse(self.lge + (self.lmw - K**2*sb*2*(self.ts - eps_e) + self.lFx)[None, :], axis=1)
        ly = self.lmw - K**2*sb*2*eps_e + self.lGss                        # hub code weight (log, unnormalised)
        keep = self.keep; lyk = ly[keep]
        self.lPh = lyk - lse(lyk); self.lwh = self.lPh - self.lrho
        self.Sbh_h = -(np.exp(self.lPh)*(self.lPh - self.lrho)).sum()
        self.lE = lse(self.lge + (self.lmw - K**2*sb*2*eps_e)[None, :], axis=1)   # emission cap inside B

    def elements_e(self, y):
        X = self.X; K = X.k; sb = X.s; eps, ee = self.eps, self.eps_e
        Pk = {0: self.logI(self.D, y, +1), 1: self.logI(self.D + self.om, y, +1)}
        Pb = {0: self.logI(self.D, y, -1), 1: self.logI(self.D + self.om, y, -1)}
        corr = -np.pi*(K[:, None] + K[None, :])
        rowB = (self.lmw - K**2*sb*(eps - 2*ee) + self.lE)[None, :]
        phys, Y, Y0 = {}, {}, {}
        lgapP = self.lmw - K**2*sb*(self.ts + eps/2 - 2*ee)
        for a, b in ((0, 0), (1, 1), (0, 1)):
            M1, M2 = Pk[a] + corr, Pb[b] + corr
            phys[(a, b)] = lse((M1 + M2 - self.lgx + self.colD + rowB)[1:, 1:])
            S1 = lse(M1 + self.colD, axis=0); S2 = lse(M2 + self.colD, axis=0)
            G1 = lse(self.lge + (lgapP + 1j*K**2*sb*y + S1)[None, :], axis=1)
            G2 = lse(self.lge + (lgapP - 1j*K**2*sb*y + S2)[None, :], axis=1)
            Y[(a, b)] = (self.lmw - K**2*sb*2*ee + G1 + G2 - self.lGss)[self.keep]
            Y0[(a, b)] = (self.lmw - K**2*sb*eps + S1 + S2 - self.lFx)[self.keep]   # Step 3h (no emission)
        return phys, Y, Y0

    def filter_e(self, x):
        if not np.isfinite(x): return np.zeros(self.lrho.size)
        try:
            R = ExactResolvent(self.lrho, [self.lPh, self.lPh], self.Sbh_h + x)
            return self.lwh + 2*R.logJ()[0]
        except RuntimeError:                       # continuation failed: canonical filter (flagged in output)
            print(f"   [x={x}: exact resolvent failed, canonical filter used]", flush=True)
            logk = self.Sbh_h + x; cum = np.logaddexp.accumulate(self.lrho); i = min(int(np.searchsorted(cum, logk)), self.lrho.size - 1)
            ll0 = lse(self.lPh[i:]) - logk if i < self.lrho.size - 1 else -np.inf
            return self.lwh - np.logaddexp(ll0, self.lwh)

def vis(r, om):
    V = np.exp(r[(0, 1)] - 0.5*(r[(0, 0)] + r[(1, 1)])); return [float(abs(V)), float(np.angle(V)/om)]

def run(C, De_list, ee_list, xs, ts=np.pi/4, D=2.0, h=1.0, eps=0.05, om=0.2, labels=('horizon', '0.9 inner')):
    x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    alllabs = {'y_H-1': yH - 1/x1, 'horizon': yH, '0.9 inner': -0.9*yH}
    labs = {k: alllabs[k] for k in labels}
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/min(ee_list)) + 10; N = int(min(2800, max(1400, kmax/0.05)))
    S = Decorated(C, ts, D, om, h, eps, kmax, N); res = []
    for ee in ee_list:
        for De in De_list:
            t0 = time.time(); S.setup_emission(De, ee)
            filt = {x: S.filter_e(x) for x in xs}
            for nm, y in labs.items():
                phys, Y, Y0 = S.elements_e(y)
                r0 = vis({k: lse(v) for k, v in Y0.items()}, om)
                row = dict(C=C, De=De, eps_e=ee, label=nm, phys=vis(phys, om), dec3h_deep=r0, dec={})
                for x in xs:
                    row['dec'][str(x)] = vis({k: lse(v + filt[x]) for k, v in Y.items()}, om)
                res.append(row)
                d = row['dec']['inf']
                print(f"C={C:4.0f} De={De:4.2f} eps_e={ee:.4f} {nm:9s}: phys {row['phys'][1]:+.5f} (|V| {row['phys'][0]:.5f}) | "
                      f"dec deep {d[1]:+.5f} (|V| {d[0]:.5f}) C*(dec-phys) {C*(d[1]-row['phys'][1]):+.4f} | 3h(no emission) deep {r0[1]:+.5f} | "
                      + " ".join(f"x={x}: {row['dec'][str(x)][1]:+.4f}" for x in xs if np.isfinite(x)) + f"  [{time.time()-t0:.0f}s]", flush=True)
    return res

if __name__ == "__main__":
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    Des = [float(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.05, 0.25, 0.5, 1.0, 2.0]
    ees = [float(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.0125, 0.02]
    xs = [0, 12, 20, np.inf]
    res = run(C, Des, ees, xs)
    json.dump(res, open(OUT + f'step4b_decorated_C{int(C)}.json', 'w'), indent=1, default=float)
