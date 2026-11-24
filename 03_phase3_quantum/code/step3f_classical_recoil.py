import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3f_ramsey as R, step3e_matter_clock as M, step3d_backreaction as B
ts=np.pi/4; dH=B.dy_horizon(ts)
for (y1,y2,nm) in ((dH,0.0,'horizon->phi0'),(dH-2,dH,'exterior->horizon')):
    dtau=M.tau_cl(ts,y2)-M.tau_cl(ts,y1); print(nm,"Dtau=",dtau)
    for (D,om,h) in ((2.,0.5,1.),(5.,0.5,1.),(20.,0.5,1.),(100.,0.5,1.),(20.,0.5,0.5),(20.,1.0,1.)):
        out=[]
        for eps in (1e-3,1e-4,1e-5):
            r,_=R.ramsey_cl(ts,y1,y2,eps,D,om,h); out.append(r)
        r=out[-1]; print(f"  D={D:5.1f} om={om} h={h}: |V|={np.exp(r.real):.6f}  -lnV*D={-r.real*D:.4f}  (phase+om*Dtau)*D={(r.imag+om*dtau)*D:+.4f}   eps-conv: "+" ".join(f"{np.exp(z.real):.6f}/{z.imag:+.6f}" for z in out))
