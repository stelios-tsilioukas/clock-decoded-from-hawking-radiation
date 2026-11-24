import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
k1 = 20.; om = 0.2
C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0); yHp = B.dy_horizon(np.pi/4)
kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/0.0125) + 10
S = Decorated(C, ts, 2.0, om, 1.0, 0.05, kmax, 2800)
labs = [-3.0, -2.0, -1.0, -0.5, yHp, -0.1, 0.1, 0.3119]
res = []
for De, ee in ((1.0, 0.0125), (1.0, 0.025), (1.0, 0.05), (2.0, 0.0125), (0.25, 0.0125)):
    S.setup_emission(De, ee); line = []
    for lab in labs:
        phys, Y, Y0 = S.elements_e(lab/x1)
        vp = vis(phys, om); vd = vis({k: lse(v) for k, v in Y.items()}, om)
        line.append((lab, k1*(vd[1] - vp[1]), 1 - vd[0], 1 - vp[0]))
    print(f"De={De} eps_e={ee}: " + "  ".join(f"[{l:+.2f}: {d:+.4f} (1-|V|d {vd:.1e})]" for l, d, vd, vp in line), flush=True)
    res.append(dict(De=De, eps_e=ee, rows=line))
json.dump(res, open('04_phase4_radiating/results/step4e_emission_scan.json', 'w'), indent=1)
