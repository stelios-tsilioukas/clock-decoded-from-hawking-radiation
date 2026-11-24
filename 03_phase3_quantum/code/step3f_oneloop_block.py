import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3f_ramsey as R, step3d_backreaction as B
ts=np.pi/4; dH=B.dy_horizon(ts); y1,y2=dH,0.0; eps=0.05; D,om,h=5.0,0.5,1.0
for a in ((y1,y2),(y1,y1),(y2,y2)):
    o=R.oneloop_block(ts,*a,eps,D,om,h); og=R.oneloop_block(ts,*a,eps,D,om,h,A=-2.,B=6.)
    print(a, "one-loop:",np.round(o,4)," gauge diff",abs(o-og), " heavy part:",np.round(R.oneloop_block(ts,*a,eps,D,om,h,heavy=True),4))
