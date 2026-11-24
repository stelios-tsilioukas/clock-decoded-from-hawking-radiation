import numpy as np, sys
sys.path.insert(0,'03_phase3_quantum/code')
from step3b_exact_rate import Exact, R_cl
from step3a_oneloop import oneloop, dtau_cl
import warnings; warnings.filterwarnings('ignore')
ts=np.pi/4
def r_oneloop(y,h=1e-3):
    V1,d2=oneloop(ts,y-h,y+h); return (-V1).real/dtau_cl(ts,y-h,y+h)**2
def shift_rate(y,h=1e-3):   # d/dy of the mean shift of Dtau, divided by R_cl : test-clock relative mean shift (C=1)
    a=oneloop(ts,-5.0,y+h)[1]/1j; b=oneloop(ts,-5.0,y-h)[1]/1j
    return ((a-b).real/(2*h))/R_cl(ts,y)
for C in (20.0,40.0):
    X=Exact(C)
    for dy in (-2.347,-1.347,-0.347,0.0,0.312):
        row=[]
        for m in (3.0,6.0):
            M=X.moments(ts,dy,m,[1,2])
            mean=M[1]/R_cl(ts,dy); relvar=M[2]/M[1]**2-1
            row.append((m,mean,relvar))
        (m1,a1,v1),(m2,a2,v2)=row
        mean0=a1-(a2-a1)/(m2-m1)*m1        # extrapolate m -> 0
        slope=(a2-a1)/(m2-m1)
        print(f"C={C:4.0f} dy={dy:+.3f}: relvar*C = {v1.real*C:.4f} (m=3), {v2.real*C:.4f} (m=6) | one-loop r = {r_oneloop(dy):.4f} || "
              f"(mean-1)*C: m=3 {((a1-1)*C).real:+.4f}, m=6 {((a2-1)*C).real:+.4f}; m->0: {((mean0-1)*C).real:+.4f}, slope/m {(slope*C).real:+.4f} | one-loop test-clock {shift_rate(dy):+.4f} | imag parts {a1.imag:.1e} {v1.imag:.1e}")
