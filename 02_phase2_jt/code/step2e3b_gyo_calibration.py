"""
Calibration C3: Gyongyosi et al.'s own protocol in our machinery.
A null energy pulse enters from the boundary at boundary time y0 (ingoing ray a = a0 = f(y0)).
Late exterior (a > a0) = phi_evap (our background, read as the post-pulse black hole).
Pre-pulse region (a < a0) = phi_evap + nuG * n_a0,  n_a0 = (a-a0)(b-a0)/(b-a)  (exact source solution of a null shell).
Prediction (their 3.32 applied before the shock, with U(u) -> -lambda):
   old horizon at bh = h, I1 QES at bh = 4h/3, ingoing lag: y0 - Y_Q = log(16 S lambda / c)/(2 kappa),  S/c = kappa/(6k).
Hat coordinates normalised at y0 (ah0 = -1).
"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
from step2e3b_release import Release, form_val_grad, ip
from scipy.optimize import brentq

class Anchor:                                   # minimal stand-in for Drop: only the hat normalisation is used
    def __init__(self, M, y0): self.lGS = M.lG(y0); self.YS = y0

class Injection(Release):
    def __init__(self, M, y0, nuG):
        self.M = M; self.D = Anchor(M, y0); self.mu = 0.0
        self.a0 = -1.0; self.nuG = nuG; self.N0 = np.array([1.0, 1.0, 1.0])   # (a0^2, -a0, 1) with a0 = -1
    def region(self, ah, bh): return 'old' if ah < self.a0 else 'new'
    def delta(self, ah, bh, reg):
        if reg == 'new': return 0.0, 0.0, 0.0
        v, ga, gb = form_val_grad(self.N0, ah, bh); return self.nuG*v, self.nuG*ga, self.nuG*gb

def local_vacuum_form(M, y0, lGS):
    """Vacuum form fitted to phi_evap (value + gradient) at the horizon point (a0, b=t_inf) in hat coords."""
    P = M.pieces(y0); w = P['w']; r = 0.0; ah = -np.exp(P['lG'] - lGS); bh = 0.0
    phi = M.phi_r(y0, r, P)
    Nga = P['n0'] + P['C1']; Na_ = -P['C1']; Nb_ = -P['C1'] + P['m2']
    dah = (w*(Na_ + Nga))/(-ah*w); dbh = (Nb_ - Nga)/(-ah)
    # solve for (A,B,C): phi, d_a phi, d_b phi of (A+B(a+b)+Cab)/(b-a) at (ah, bh)
    a, b = ah, bh
    def vg(Z): return np.array(form_val_grad(Z, a, b))
    Mx = np.column_stack([vg(np.eye(3)[i]) for i in range(3)])
    return np.linalg.solve(Mx, np.array([phi, dah, dbh]))

if __name__ == "__main__":
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
    M = LogJT(k); y0 = 0.5/k; kap = M.kappa(y0)
    lGS = M.lG(y0); Znew = local_vacuum_form(M, y0, lGS)
    print(f"k={k} y0={y0} kappa={kap:.4f}  local Z_new={Znew}", flush=True)
    N0 = np.array([1., 1., 1.])
    bgrid = np.sort(np.concatenate([np.geomspace(1e-12, 0.6, 500), -np.geomspace(1e-12, 0.6, 300)]))
    rows = []
    for lam_t in (0.003, 0.01, 0.03):
        # choose nuG so that lambda = (phi_h,new - phi_h,old)/(phi_h,new + phi_h,old) = lam_t
        def lam_of(nu):
            pn = np.sqrt(-ip(Znew, Znew)); po = np.sqrt(-ip(Znew + nu*N0, Znew + nu*N0)); return (pn - po)/(pn + po)
        nuG = brentq(lambda nu: lam_of(nu) - lam_t, 1e-9, 0.5)
        Zold = Znew + nuG*N0
        roots = np.roots([Zold[2], 2*Zold[1], Zold[0]]); h = roots[np.argmin(np.abs(roots))].real   # old future horizon
        I = Injection(M, y0, nuG)
        SmS0_c = kap/(6*k); lag_pred = np.log(16*SmS0_c*lam_t)/(2*kap)
        for dy in (3.0, 6.0, 9.0):
            yB = y0 + dy/kap
            found = I.find(yB, 'old', np.linspace(y0 - 4*lag_pred - 2, y0 - 1e-3, 81), bgrid)
            Yq, eta = M.qes(yB); bq = eta*np.exp(M.lG(yB) - lGS); Sstd = I.Sgen(Yq, bq, yB, 'new')
            s = f"lam={lam_t} yB-y0={dy/kap:.2f}: h={h:.4e} pred bh_Q=4h/3={4*h/3:.4e}, lag_pred={lag_pred:.3f}"
            for q in found:
                s += f" | I1: bh={q['bh']:.4e} (ratio {q['bh']/(4*h/3):.3f}) lag={y0-q['Y']:.3f} (ratio {(y0-q['Y'])/lag_pred:.3f}) S-Sstd={q['S']-Sstd:+.3e}"
            if not found: s += " | no I1 found"
            print(s, flush=True)
            rows.append(dict(lam=lam_t, nuG=nuG, h=h, dy=dy, lag_pred=lag_pred, found=found, Sstd=Sstd))
    json.dump(rows, open(f'02_phase2_jt/results/step2e3b_gyo_calibration_k{k:g}.json', 'w'),
              indent=1, default=float)
