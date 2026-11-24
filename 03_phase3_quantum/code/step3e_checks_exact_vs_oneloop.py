import numpy as np, sys, warnings, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3e_matter_clock as M
ts=np.pi/4; dys=np.array([-2.347,-0.347,0.0,0.312])
for (D,om,h) in ((2.0,0.0,1.0),(2.0,0.5,1.0),(1.0,0.3,0.8),(0.6,0.2,0.5)):
    wa,wb,wc=M.weights(D,om,h)
    ol=np.array([M.oneloop_triangle(ts,d,wa,wb,wc) for d in dys])
    print(f"Delta={D} omega={om} h={h}  weights=({wa:.2f},{wb:.2f},{wc:.2f})  one-loop: "+" ".join(f"{z.real:+.4f}{z.imag:+.4f}i" for z in ol))
    for C in (20.,40.,80.):
        t=time.time(); X=M.Exact(C,N=4000); ex=X.log_triangle(ts,dys,wa,wb,wc); cl=M.logA_cl(ts,dys,wa,wb,wc)
        z=C*(ex-cl); print(f"   C={C:3.0f}: C(ex-cl) = "+" ".join(f"{v.real:+.4f}{v.imag:+.4f}i" for v in z)+f"  ({time.time()-t:.0f}s)",flush=True)
