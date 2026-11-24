# Step 2E.3b calibration run: tossed (boundary-injection-like) clock, near-null tau_s=pi/16. Usage: python this.py 1e-4 0.0625
import numpy as np, sys
sys.path.insert(0,'02_phase2_jt/code')
from step2_logjt import LogJT
from step2d_clock import Drop, gd
from step2e_qes_kink import KinkQES, lam_gyongyosi
from scipy.optimize import brentq
k=float(sys.argv[1]); ts=float(sys.argv[2])*np.pi
M=LogJT(k); ydrop=0.5/k
kap=M.kappa(ydrop); phi_s=2*kap/np.sin(ts)
YS=ydrop-0.5
for _ in range(40):
    D=Drop(M,YS,phi_s); YS+=ydrop-D.y_radar
D=Drop(M,YS,phi_s); YH=D.Y_of_ah(D.point(D.lamH)[0])
Dl=np.log(4*kap/k)/(2*kap)
print(f'k={k} ts={ts:.4f} kap={kap:.4f} YS-YH={YS-YH:.3f} Delta~{Dl:.3f}')
for lam_t in (0.003,0.01,0.03):
    mu=4*kap*lam_t*np.tan(ts); Q=KinkQES(M,D,mu); lam,eps=lam_gyongyosi(D,M,mu)
    for dy in (0.0, 2.0):
        yB=YH+Dl+dy
        # bracket scan in Y; inner 1D in bh
        def bstar(Y):
            ah=-np.exp(M.lG(Y)-D.lGS); bC1=D.t0+D.R**2/(D.t0-ah)
            bs=np.concatenate([bC1+np.geomspace(1e-13,2,300), np.geomspace(1e-12,0.5,400), -np.geomspace(1e-12,0.5,400)])
            bs=np.sort(bs[bs>bC1]); g=np.array([Q.grad(Y,b,yB,'in')[1] for b in bs])
            idx=np.where(np.sign(g[:-1])!=np.sign(g[1:]))[0]
            res=[brentq(lambda b:Q.grad(Y,b,yB,'in')[1],bs[i],bs[i+1],xtol=1e-18) for i in idx]
            res=[b for b in res if b>0 and abs(Q.grad(Y,b,yB,'in')[1])<1e-6]   # genuine roots inside the event horizon
            return res[-1] if res else np.nan
        Ys=np.linspace(YH-16,YH,129); gY=[]
        for Y in Ys:
            b=bstar(Y); gY.append(Q.grad(Y,b,yB,'in')[0] if np.isfinite(b) else np.nan)
        gY=np.array(gY); idx=[i for i in range(96) if np.isfinite(gY[i]) and np.isfinite(gY[i+1]) and np.sign(gY[i])!=np.sign(gY[i+1])]
        Yq,eta=M.qes(yB); bq=eta*np.exp(M.lG(yB)-D.lGS); Sout=Q.Sgen(Yq,bq,yB,'out')
        pout=Q.p_and_grad(-np.exp(M.lG(Yq)-D.lGS),bq)[0]
        for i in idx:
            Y=brentq(lambda Y: Q.grad(Y,bstar(Y),yB,'in')[0],Ys[i],Ys[i+1],xtol=1e-11); b=bstar(Y)
            ah=-np.exp(M.lG(Y)-D.lGS); p=Q.p_and_grad(ah,b)[0]; Sin=Q.Sgen(Y,b,yB,'in')
            tau_b=Q.tau_on_C1_b(b)
            print(f'  lam={lam:.4f} yB-YH-Delta={dy}: in-QES Y-YH={Y-YH:+.3f} (Y-YS={Y-YS:+.3f}) bh={b:+.3e} p={p:+.2e} '
                  f'S_in-S_out={Sin-Sout:+.3e} (out p={pout:+.1e}) | tau_end={tau_b:.5f}  8lam/3 tan={8*lam/3*np.tan(ts):.5f} ratio={tau_b/(8*lam/3*np.tan(ts)):.3f}'
                  f'  lag (YH-Y)*2kap={2*kap*(YH-Y):.3f} vs log(dS/c)={np.log(lam*kap/(3*k)):.3f}')
