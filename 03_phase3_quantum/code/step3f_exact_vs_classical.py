import numpy as np, sys, time, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3f_ramsey as R, step3d_backreaction as B
ts=np.pi/4; dH=B.dy_horizon(ts); y1,y2=dH,0.0; eps=0.05
D,om,h=5.0,0.5,1.0
lc12,_=R.logG_cl(ts,y1,y2,eps,D,om,h); lc11,_=R.logG_cl(ts,y1,y1,eps,D,om,h); lc22,_=R.logG_cl(ts,y2,y2,eps,D,om,h)
print("classical logs:",lc12,lc11,lc22)
for C,kmax,N in ((5.,40.,1400),(10.,60.,1800),(20.,90.,2400)):
    t=time.time(); X=R.ExactRamsey(C,kmax,N)
    r,(l12,l11,l22)=X.ramsey(ts,y1,y2,eps,D,om,h)
    print(f"C={C}: C*(exact-cl): F12 {C*(l12-lc12):.4f}  F11 {C*(l11-lc11):.4f}  F22 {C*(l22-lc22):.4f}  | ratio: C*(lnV_ex - lnV_cl) = {C*(r-(lc12-(lc11+lc22)/2)):.4f}  ({time.time()-t:.0f}s)",flush=True)
