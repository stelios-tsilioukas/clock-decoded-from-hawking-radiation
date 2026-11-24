import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
k1 = float(sys.argv[1]) if len(sys.argv) > 1 else 20.
om = 0.2
C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0); yHp = B.dy_horizon(np.pi/4)
kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/0.0125) + 10
S = Decorated(C, ts, 2.0, om, 1.0, 0.05, kmax, 2800); S.setup_emission(1.0, 0.0125)
X = S.X; K = X.k; corr = -np.pi*(K[:, None] + K[None, :])
rowB = S.lmu + S.lw - K**2*X.s*S.eps
labs = {'y_H-2': yHp-2, 'y_H-1': yHp-1, 'y_H-0.5': yHp-0.5, 'horizon': yHp, 'phi=0': 0.0, '0.9 inner': -0.9*yHp}
print(f"k1={k1} C={C:.3f} kappa1={x1:.4f}")
for nm, lab in labs.items():
    y = lab/x1
    M1 = S.logI(S.D, y, +1) + corr; M2 = S.logI(S.D + S.om, y, -1) + corr       # ket level 0, bra level 1 (rho_01)
    lp = S.lgx + S.colD - S.lFx[None, :]                                        # log p(D|s)
    lf1, lf2 = M1 - S.lgx, M2 - S.lgx
    l_joint = lse(lp + lf1 + lf2, axis=0); l_fac = lse(lp + lf1, axis=0) + lse(lp + lf2, axis=0)
    w = rowB + S.lFx
    ph = lse((w + l_joint)[1:]); de = lse((w + l_fac)[1:])
    r_phys = np.angle(np.exp(ph - ph.real))/om; r_dec = np.angle(np.exp(de - de.real))/om
    # diagnostics at the dominant s
    js = np.argmax((w + l_joint.real)[1:]) + 1
    p = np.exp(lp[:, js] - lse(lp[:, js]).real).real
    phiD = np.unwrap(np.angle(np.exp(lf1[:, js] + lf2[:, js])))/om                 # phase of the integrand vs D
    ellD = (lf1[:, js] + lf2[:, js]).real
    mD = (p*K).sum(); vD = (p*(K - mD)**2).sum()
    mphi = (p*phiD).sum(); vphi = (p*(phiD - mphi)**2).sum()
    cov_phi_ell = (p*(phiD - mphi)*(ellD - (p*ellD).sum())).sum()
    sl = np.polyfit(K[p > 1e-6*p.max()], phiD[p > 1e-6*p.max()], 1, w=np.sqrt(p[p > 1e-6*p.max()]))[0]
    print(f"{nm:9s} label {lab:+.3f}: reading phys {r_phys:+.5f} decoded(3h, deep) {r_dec:+.5f}  dec-phys {r_dec-r_phys:+.6f} | "
          f"s*={K[js]:.2f}  <D>={mD:.2f}  sd(D)={np.sqrt(vD):.3f}  dphase/dD={sl:+.5f}  sd(phase)={np.sqrt(vphi):.5f}  cov(phase,log-amp)={cov_phi_ell:+.6f}", flush=True)
