import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3e_matter_clock as M, step3d_backreaction as B
from step3a_oneloop import oneloop as ol3a
ts=np.pi/4
# consistency: polynomial reproduces direct one-loop incl. normalisation
D,om,h=1.3,0.4,0.9; dy=-0.5
wa,wb,wc=M.weights(D,om,h)
direct=M.oneloop_triangle(ts,dy,wa,wb,wc)-0.5*M.oneloop_two_point(D,2*ts)-0.5*M.oneloop_two_point(D+om,2*ts)
print("poly vs direct:",M.lnZhat_1loop(ts,dy,D,om,h),direct)
dH=B.dy_horizon(ts)
labs=[('y_H-2',dH-2),('y_H-1',dH-1),('horizon',dH),('phi=0',0.0),('0.9 inner',-0.9*dH)]
for nm,dy in labs:
    c=M.lnZhat_coeffs(ts,dy)
    print(nm, " ".join(f"{k}:{v.real:+.4f}{v.imag:+.4f}i" for k,v in c.items()))
# 3d prediction for the D*w coefficient: kappa1' * d tau0/d kappa
k1p=B.kappa1_lin(ts)
tau0=lambda dy,k: 2*np.arctan(np.tan(k*ts/2)**-1*np.tanh(k*dy/2))
for nm,dy in labs:
    pred=k1p*(tau0(dy,1+1e-6)-tau0(dy,1-1e-6))/2e-6
    print(nm,"Im Dw =",M.lnZhat_coeffs(ts,dy)['Dw'].imag," 3d prediction",pred)
# mean reading shift differences vs 3a interval mean shift
for (y1,y2) in ((dH,0.0),(dH-2,dH),(-1.0,0.8*np.log(1/np.cos(ts)))):
    s=M.lnZhat_coeffs(ts,y2)['w'].imag-M.lnZhat_coeffs(ts,y1)['w'].imag
    V1,d2=ol3a(ts,y1,y2); print(f"interval {y1:+.3f}->{y2:+.3f}: single-event difference {s:+.5f}  3a mean shift {(d2/1j).real:+.5f} (im {(d2/1j).imag:+.1e})")
