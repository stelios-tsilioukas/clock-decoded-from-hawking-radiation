import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3f_ramsey as R, step3d_backreaction as B
tsp = np.pi/4; dH = B.dy_horizon(tsp); y1, y2 = dH, 0.0; epsp = 0.05
for k1, D, om, h in ((10, 2., 0.5, 1.), (4, 1.5, 0.3, 0.5)):
    C, ts, x1 = B.phys_to_bare(k1, tsp, D); kmax0 = max(1.8*C + 8*np.sqrt(C) + 10, 3*np.sqrt(2*C*x1/epsp))
    for fk, fN in ((1.0, 1.0), (1.3, 1.0), (1.3, 1.6)):
        kmax = kmax0*fk; N = int(min(3200, max(1200, kmax0/0.045))*fN*fk)
        lv, _ = R.ExactRamsey(C, kmax, N).ramsey(ts, y1/x1, y2/x1, epsp/x1, D, om, h)
        print(f"k1={k1} D={D}: kmax={kmax:.1f} N={N}: ln V_exact = {lv.real:+.8f}{lv.imag:+.8f}i", flush=True)
