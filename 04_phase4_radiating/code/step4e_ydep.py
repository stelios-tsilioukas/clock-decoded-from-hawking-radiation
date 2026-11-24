import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
from step3k_split import reading_cl
k1 = float(sys.argv[1]); om = 0.2
C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0); yHp = B.dy_horizon(np.pi/4)
kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/0.0125) + 10
S = Decorated(C, ts, 2.0, om, 1.0, 0.05, kmax, 2800); S.setup_emission(1.0, 0.0125)
out = []
for lab in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, yHp, -0.2, -0.1, 0.1, 0.2, 0.3119]:
    phys, Y, Y0 = S.elements_e(lab/x1)
    rp = vis(phys, om)[1]; rd = vis({k: lse(v) for k, v in Y.items()}, om)[1]; r3 = vis({k: lse(v) for k, v in Y0.items()}, om)[1]
    rc = reading_cl(np.pi/4, lab, 2.0, om, 1.0, x1*0.05)[1]
    print(f"label {lab:+.4f}: classical {rc:+.5f} one-sided {rp:+.6f} | k1*(dec4b - phys) {k1*(rd-rp):+.5f}  k1*(dec3h - phys) {k1*(r3-rp):+.5f}  k1*(dec4b - dec3h) {k1*(rd-r3):+.5f} | k1*(phys - cl) {k1*(rp-rc):+.5f}", flush=True)
    out.append(dict(label=lab, classical=rc, phys=rp, dec4b=rd, dec3h=r3))
json.dump(out, open(f'04_phase4_radiating/results/step4e_ydep_k{int(k1)}.json', 'w'), indent=1)
