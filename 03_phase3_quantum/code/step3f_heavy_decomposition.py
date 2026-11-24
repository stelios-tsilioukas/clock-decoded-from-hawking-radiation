import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3f_ramsey as R, step3d_backreaction as B, step3e_matter_clock as M
from step3a_oneloop import oneloop as ol3a
for ts in (np.pi/4, 3*np.pi/8):
    dH=B.dy_horizon(ts)
    for (y1,y2,nm) in ((dH,0.0,'horizon->phi0'),(dH-2,dH,'y_H-2->horizon'),(dH,-0.9*dH,'horizon->0.9inner')):
        for eps in (0.0,0.05):
            c,res=R.lnV_poly(ts,y1,y2,eps)
            print(f"ts={ts:.3f} {nm} eps={eps} (fit resid {res:.1e}): "+"  ".join(f"{k}={v.real:+.4f}{v.imag:+.4f}i" for k,v in c.items()))
        V1,d2=ol3a(ts,y1,y2); print(f"   3a worldline: Var_K={(-V1).real:.4f} commutator={(-V1).imag:+.4f} mean shift={(d2/1j).real:+.4f}")
        k1p=B.kappa1_lin(ts); dt=lambda k: (lambda f: f(y2)-f(y1))(lambda y: 2*np.arctan(np.tanh(k*y/2)/np.tan(k*ts/2)))
        print(f"   3d backreaction prediction for Im(Dw): -kappa1'*d(Dtau)/dkappa = {-k1p*(dt(1+1e-6)-dt(1-1e-6))/2e-6:+.4f}")
