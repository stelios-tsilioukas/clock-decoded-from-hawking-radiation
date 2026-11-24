"""Calibration C3 (exact): Gyongyosi protocol with the exactly constructed two-pulse background (step2_logjt_pulse.py).
Search the I1 QES in the pre-pulse region (Y < y0) and compare with their (3.32)-based prediction:
   bh_Q = (4/3) h,  h = old-horizon offset ~ lambda (hat units, G(y0) = 1);  lag y0 - Y_Q = log(16 S lambda/c)/(2 kappa)."""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt_pulse import LogJTPulse
from step2e3b_release import Release
from scipy.optimize import brentq

class Anchor:
    def __init__(self, M, y0): self.lGS = M.lG(y0); self.YS = y0
class Exact(Release):
    def __init__(self, M, y0):
        self.M = M; self.D = Anchor(M, y0); self.mu = 0.0; self.y0 = y0
    def region(self, ah, bh): return 'old' if ah < -1.0 else 'new'
    def delta(self, ah, bh, reg): return 0.0, 0.0, 0.0

k = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
y0 = 0.5/k; rows = []
bgrid = np.sort(np.concatenate([np.geomspace(1e-12, 0.6, 600), -np.geomspace(1e-12, 0.6, 300)]))
for lam in (0.003, 0.01, 0.03):
    M = LogJTPulse(k, y0, lam); E = Exact(M, y0)
    kap_old = np.sqrt(M.V(y0 - 1e-9)); kap_new = np.sqrt(M.V(y0 + 1e-9))
    # old-horizon offset h: zero of d_a phi along a = a0^- at late b ~ 0 (apparent horizon of the old dilaton at a0)
    P = M.pieces(y0 - 1e-9)
    rAH = -P['n0']/(P['C1'] + np.sqrt(P['C1']**2 + P['m2']*P['n0']))   # r = g_b/g_a at the apparent horizon
    h_AH = -rAH                                                       # bh = r*ah with ah = -1
    S_c = kap_old/(6*k); lag_pred = np.log(16*S_c*lam)/(2*kap_old)
    for dy in (4.0, 8.0, 12.0):
        yB = y0 + dy
        found = E.find(yB, 'old', np.linspace(y0 - 3*lag_pred - 3, y0 - 1e-3, 91), bgrid)
        s = (f"lam={lam} yB-y0={dy}: kap_old={kap_old:.4f} kap_new={kap_new:.4f} h_AH(old,at a0)={h_AH:.4e} "
             f"pred bh=(4/3)lam={4*lam/3:.4e} lag_pred={lag_pred:.3f}")
        for q in found:
            s += f" | QES: bh={q['bh']:.4e} (bh/(4lam/3)={q['bh']/(4*lam/3):.3f}) lag={y0-q['Y']:.3f} (ratio {(y0-q['Y'])/lag_pred:.3f}) S={q['S']:.6f}"
        if not found: s += " | none"
        print(s, flush=True)
        rows.append(dict(lam=lam, dy=dy, h_AH=h_AH, lag_pred=lag_pred, found=found))
json.dump(rows, open(f'02_phase2_jt/results/step2e3b_gyo_exact_k{k:g}.json', 'w'), indent=1, default=float)
