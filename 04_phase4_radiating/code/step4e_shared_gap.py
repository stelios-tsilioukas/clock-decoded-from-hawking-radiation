import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
k1 = 20.; om = 0.2
C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0)
kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/0.0125) + 10
S = Decorated(C, ts, 2.0, om, 1.0, 0.05, kmax, 2800); S.setup_emission(1.0, 0.0125)
X = S.X; K = X.k; sb = X.s; eps, ee = S.eps, S.eps_e
def variant(y, gap_phase=True, shared_gap=False):
    Pk = {0: S.logI(S.D, y, +1), 1: S.logI(S.D + S.om, y, +1)}; Pb = {0: S.logI(S.D, y, -1), 1: S.logI(S.D + S.om, y, -1)}
    corr = -np.pi*(K[:, None] + K[None, :]); lgapP = S.lmw - K**2*sb*(S.ts + eps/2 - 2*ee); Y = {}
    for a, b in ((0, 0), (1, 1), (0, 1)):
        M1, M2 = Pk[a] + corr, Pb[b] + corr
        S1 = lse(M1 + S.colD, axis=0); S2 = lse(M2 + S.colD, axis=0)
        ph = 1j*K**2*sb*y if gap_phase else 0.0
        if shared_gap:   # ket and bra share one gap energy g (as in the one-sided state), weight squared lgapP -> one factor
            lA = S.lmw - K**2*sb*(S.ts + eps/2 - 2*ee); lAp = S.lmw - K**2*sb*2*(S.ts - ee)
            term = lA + 1j*K**2*sb*y + lA - 1j*K**2*sb*y + S1 + S2 - lAp - S.lFx
            term = np.where(np.isfinite(S.lFx) & np.isfinite(lAp), term, -np.inf)
            G = lse(S.lge + term[None, :], axis=1)
            Y[(a, b)] = (S.lmw - K**2*sb*2*ee + G)[S.keep]
        else:
            G1 = lse(S.lge + (lgapP + ph + S1)[None, :], axis=1); G2 = lse(S.lge + (lgapP - ph + S2)[None, :], axis=1)
            Y[(a, b)] = (S.lmw - K**2*sb*2*ee + G1 + G2 - S.lGss)[S.keep]
    return vis({k: lse(v) for k, v in Y.items()}, om)[1]
for lab in (-3.0, -2.0, -1.0, -0.35, 0.31):
    y = lab/x1; phys, Yf, _ = S.elements_e(y); rp = vis(phys, om)[1]; rd = vis({k: lse(v) for k, v in Yf.items()}, om)[1]
    r_shared = variant(y, shared_gap=True)
    print(f"label {lab:+.2f}: k1*(dec - phys) full {k1*(rd-rp):+.4f} | gap shared by ket & bra {k1*(r_shared-rp):+.4f}", flush=True)
