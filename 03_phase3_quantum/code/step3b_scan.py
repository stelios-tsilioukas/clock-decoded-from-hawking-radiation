import numpy as np, sys
sys.path.insert(0,'03_phase3_quantum/code')
from step3b_exact_rate import Exact, R_cl
import warnings; warnings.filterwarnings('ignore')
ts=np.pi/4
pos=[('y_H-2',-2.347),('y_H-1',-1.347),('horizon',-0.347),('phi=0',0.0),('0.9 inner',0.312)]
for m in (2.0,4.0):
    print(f"--- m = {m}")
    for C in (40.0,10.0,4.0,2.0,1.0,0.5,0.25,0.1):
        X=Exact(C); s=f"C={C:5.2f} S-S0={2*np.pi*C:6.2f}:"
        for name,dy in pos:
            M=X.moments(ts,dy,m,[1,2,3])
            mean=(M[1]/R_cl(ts,dy)).real; rv=(M[2]/M[1]**2-1).real
            k3=(M[3]-3*M[2]*M[1]+2*M[1]**3).real; skew=k3/((M[2]-M[1]**2).real)**1.5
            s+=f"  {name}: <R>/Rcl={mean:.3f} sd/mean={np.sqrt(rv):.3f} skew={skew:+.2f} |"
        print(s,flush=True)
