import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
k1=20.; C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0); yHp = B.dy_horizon(np.pi/4)
kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/0.0125) + 10
S = Decorated(C, ts, 2.0, 0.2, 1.0, 0.05, kmax, 2800)
for De, ee in ((1.0, 0.0125), (2.0, 0.0125), (1.0, 0.02), (2.0, 0.02)):
    S.setup_emission(De, ee); r = {}
    for nm, lab in (('y_H-1', yHp-1), ('horizon', yHp), ('0.9 inner', -0.9*yHp)):
        phys, Y, _ = S.elements_e(lab/x1); r[nm] = (vis(phys, .2)[1], vis({k: lse(v) for k, v in Y.items()}, .2)[1])
    ext = [r['horizon'][i]-r['y_H-1'][i] for i in (0,1)]; inn = [r['0.9 inner'][i]-r['horizon'][i] for i in (0,1)]
    print(f"De={De} eps_e={ee}: exterior interval phys {ext[0]:+.6f} dec {ext[1]:+.6f} (k1*(dec-phys)/cl = {k1*(ext[1]-ext[0])/1.12689:+.4f}) | interior phys {inn[0]:+.6f} dec {inn[1]:+.6f} ({k1*(inn[1]-inn[0])/1.50186:+.4f})", flush=True)
