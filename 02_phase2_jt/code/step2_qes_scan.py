"""Step 2C scan: QES lag and position vs k (down to 1e-4) using the overflow-free LogJT."""
import numpy as np, json, sys
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
rows = []
for k in (0.05, 0.02, 0.01, 3e-3, 1e-3, 3e-4, 1e-4):
    L = LogJT(k)
    for x in (0.3, 1.0, 2.0):                       # yB = x/k  -> kappa = e^{-x/2}
        yB = x/k; res = L.qes(yB)
        if res is None: rows.append(dict(k=k, yB=yB, found=False)); continue
        Y, eta = res; kap = L.kappa(yB); D = yB - Y
        SmS0_over_c = 4*np.pi*kap/(24*np.pi*k)         # (S-S0)/c = kappa/(6k)
        rAH = L.r_apparent(Y); q = np.exp(L.lG(yB) - L.lG(Y)); eta_AH = -rAH/q
        rows.append(dict(k=k, yB=yB, kappa=kap, Y=Y, Delta=D, kappaDelta=kap*D,
                         ammz_log=np.log(SmS0_over_c)/(2*kap),
                         offset=D - np.log(SmS0_over_c)/(2*kap),
                         offset_times_2kappa=2*kap*(D - np.log(SmS0_over_c)/(2*kap)),
                         eta=eta, eta_AH=eta_AH, found=True))
        r = rows[-1]
        print(f"k={k:<7g} yB={yB:9.2f} kap={kap:.3f} Delta={D:8.4f} (1/2k)log((S-S0)/c)={r['ammz_log']:8.4f} "
              f"2kap*offset={r['offset_times_2kappa']:.4f}  eta_QES={eta:+.4f}  eta_AH={eta_AH:+.4f}", flush=True)
json.dump(rows, open('02_phase2_jt/results/step2_qes_scan.json', 'w'), indent=1, default=float)
