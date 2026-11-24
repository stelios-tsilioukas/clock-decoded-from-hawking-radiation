"""Step 2E.3b: QES candidates for the localised-support release protocol in the evaporating background.
Usage: python step2e3b_qes.py k ts_over_pi"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
from step2d_clock import Drop, gd
from step2e3b_release import Release, ConsistentRelease
k = float(sys.argv[1]); ts = float(sys.argv[2])*np.pi
M = LogJT(k); ydrop = 0.5/k; kap = M.kappa(ydrop); phi_s = 2*kap/np.sin(ts)
YS = ydrop - 0.5
for _ in range(40):
    D = Drop(M, YS, phi_s); YS += ydrop - D.y_radar
D = Drop(M, YS, phi_s); YH = D.Y_of_ah(D.point(D.lamH)[0])
Dl = np.log(4*kap/k)/(2*kap)
print(f"k={k} ts={ts:.4f} kap={kap:.4f} Y_S-Y_H={YS-YH:.3f} Delta~{Dl:.3f}", flush=True)
bgrid = np.sort(np.concatenate([np.geomspace(1e-12, 0.6, 500), -np.geomspace(1e-12, 0.6, 300)]))
rows = []
for lam_t in (0.003, 0.01, 0.03):
    mu = 8*kap*lam_t*np.tan(ts); R = ConsistentRelease(M, D, mu, phi_s); print(f"  [lam={lam_t}] iterations={len(R.hist)} dlamS={R.dlamS:+.3e} nu={R.nu:+.4e} nup={R.nup:+.4e} tauH shift={R.tauH-D.tauH:+.3e}", flush=True)
    for dy in (-1.0, 0.0, 1.0, 3.0):
        yB = YH + Dl + dy
        Yq, eta = M.qes(yB); bq = eta*np.exp(M.lG(yB) - D.lGS); aq = -np.exp(M.lG(Yq) - D.lGS)
        std = dict(Y=Yq, bh=bq, reg=R.region(aq, bq), S=R.Sgen(Yq, bq, yB, R.region(aq, bq)))
        old = R.find(yB, 'old', np.linspace(YH - 10, YS - 1e-3, 81), bgrid)
        r2 = R.find(yB, 'R2', np.linspace(YS + 1e-3, YH + 1.0, 41), bgrid)
        row = dict(lam=lam_t, mu=mu, nu=R.nu, nup=R.nup, yB_minus_YH_minus_Delta=dy,
                   standard=std, old=old, R2=r2)
        rows.append(row)
        s = f" lam={lam_t} dy={dy:+.1f}: standard QES reg={std['reg']} S={std['S']:.8f}"
        for q in old:
            te = R.tau_C1_at_b(q['bh'])
            s += (f" | OLD-region QES Y-YH={q['Y']-YH:+.3f} bh={q['bh']:+.3e} S-Sstd={q['S']-std['S']:+.3e}"
                  f" tau_end={te:.5f} (8lam/3)tan={8*lam_t/3*np.tan(ts):.5f} ratio={te/(8*lam_t/3*np.tan(ts)):.3f}")
            q['tau_end'] = te
        for q in r2: s += f" | R2 QES Y-YH={q['Y']-YH:+.3f} bh={q['bh']:+.3e} S-Sstd={q['S']-std['S']:+.3e}"
        print(s, flush=True)
json.dump(rows, open(f'02_phase2_jt/results/step2e3b_qes_k{k:g}_ts{sys.argv[2]}.json', 'w'),
          indent=1, default=float)
