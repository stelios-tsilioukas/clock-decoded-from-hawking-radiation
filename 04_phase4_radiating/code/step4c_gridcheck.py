import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
k1=20.; C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0); yHp = B.dy_horizon(np.pi/4)
kmax0 = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/0.0125) + 10
for N, kmax in ((2000, kmax0), (2800, kmax0), (2800, 1.25*kmax0)):
    S = Decorated(C, ts, 2.0, 0.2, 1.0, 0.05, kmax, N); S.setup_emission(1.0, 0.0125)
    r = {}
    for nm, lab in (('y_H-1', yHp-1), ('horizon', yHp)):
        phys, Y, _ = S.elements_e(lab/x1); r[nm] = (vis(phys, .2)[1], vis({k: lse(v) for k, v in Y.items()}, .2)[1])
    print(f"N={N} kmax={kmax:.1f}: y_H-1 phys {r['y_H-1'][0]:+.6f} dec {r['y_H-1'][1]:+.6f} | horizon phys {r['horizon'][0]:+.6f} dec {r['horizon'][1]:+.6f} | interval phys {r['horizon'][0]-r['y_H-1'][0]:+.6f} dec {r['horizon'][1]-r['y_H-1'][1]:+.6f}", flush=True)
