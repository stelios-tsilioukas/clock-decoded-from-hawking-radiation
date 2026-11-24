import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3e_matter_clock as M
ts=np.pi/4
# (1) classical phase = omega * proper time
dys=np.linspace(-3,0.5,8)
for om in (0.3,1.0):
    wa,wb,wc=M.weights(2.0,om,1.5); ph=np.imag(M.logA_cl(ts,dys,wa,wb,wc))
    print("phase/omega - (tau - tau(0)):", np.max(np.abs(ph/om-(M.tau_cl(ts,dys)-M.tau_cl(ts,0.0)))))
# |A_cl| vs R^h
from step3d_backreaction import R_cl
wa,wb,wc=M.weights(2.0,0.4,1.3); r=np.real(M.logA_cl(ts,dys,wa,wb,wc))-1.3*np.log(R_cl(ts,dys)); print("log|A| - h log R spread:", r.max()-r.min())
# (2) gauge invariance of one-loop, general weights
for args in ((ts,-0.7,0.3,0.9,1.7),(ts,0.2,0.6,0.6,1.1)):
    a=M.oneloop_triangle(*args); b=M.oneloop_triangle(*args,A=-3.1,B=7.7); print("gauge:",a,abs(a-b))
print("2pt one-loop gauge:", M.oneloop_two_point(1.3,1.1), abs(M.oneloop_two_point(1.3,1.1)-M.oneloop_two_point(1.3,1.1,A=4,B=-2)))
# (3) two-point: exact vs classical vs one loop
for C in (20.,40.,80.):
    X=M.Exact(C,N=4000)
    for l,u in ((1.0,2.0),(2.5,0.7)):
        ex=X.log_two_point(l,u); cl=-2*l*np.log(2*np.sin(u/2)); print(f"C={C} l={l} u={u}: C(ex-cl)={C*(ex-cl).real:.4f}  one-loop={M.oneloop_two_point(l,u).real:.4f}")
