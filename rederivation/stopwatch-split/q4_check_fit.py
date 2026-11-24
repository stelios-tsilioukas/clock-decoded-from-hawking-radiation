"""Grid/tail checks for the small-h points of question 4, then the fit c0 + c_w w + c_h h."""
import numpy as np, json
from sw import exact_F, R_cl, phys_to_bare, yH_of
from oneloop import a_coeff
k1, taup, m = 20.0, np.pi/4, 2.0
d = phys_to_bare(k1, taup, m); C, k, tau = d['C'], d['kappa1'], d['tau']
lab = yH_of(taup); y = lab/k
pts = [(0.2,1,0.05),(0.1,1,0.05),(0.05,1,0.05),(0.1,0.5,0.05),(0.05,0.25,0.05),(0.2,1,0.025)]
grids = {1.0: [(0.1,None),(0.05,None),(0.05,C+14*np.sqrt(C)+330)],
         0.5: [(0.05,None),(0.035,None),(0.05,C+14*np.sqrt(C)+330)],
         0.25:[(0.03,C+9*np.sqrt(C)+3*m+50),(0.025,C+9*np.sqrt(C)+3*m+120),(0.03,C+9*np.sqrt(C)+3*m+200)]}
rows=[]
for (om,h,ep) in pts:
    eb = ep/k
    cl = R_cl(taup, lab, ep)
    rd=[]
    for dk,kxb in grids[h]:
        r = exact_F(C, tau, y, eb, m, om, h, dk=dk, kmaxB=kxb)
        rd.append(r['reading'])
    best = rd[1]
    obs = k1*(best-cl)/abs(cl)
    a = a_coeff(tau, y, eb, m, om, h)
    pt = k1*(R_cl(tau,y,eb)+a/C-cl)/abs(cl)
    rows.append(dict(omega=om,h=h,eps=ep,readings=rd,reading=best,cl=cl,obs=obs,obs_spread=k1*(max(rd)-min(rd))/abs(cl),
                     absV=r['absV'],PT_a=a,PT_obs=pt))
    print(rows[-1], flush=True)
X = np.array([[1,r['omega'],r['h']] for r in rows]); Y=np.array([r['obs'] for r in rows]); Yp=np.array([r['PT_obs'] for r in rows])
coef,res,_,_ = np.linalg.lstsq(X,Y,rcond=None); coefp,_,_,_=np.linalg.lstsq(X,Yp,rcond=None)
print('fit exact c0,cw,ch =',coef,' residuals',Y-X@coef)
print('fit PT    c0,cw,ch =',coefp,' residuals',Yp-X@coefp)
# fit only the eps=0.05 points
m5 = np.array([r['eps']==0.05 for r in rows])
c5,_,_,_=np.linalg.lstsq(X[m5],Y[m5],rcond=None); print('fit (eps=0.05 only)',c5, Y[m5]-X[m5]@c5)
# one-loop mean proper-time shift, omega,h -> 0
out=dict(rows=rows,fit=coef.tolist(),fit_resid=(Y-X@coef).tolist(),fitPT=coefp.tolist(),fit_eps005=c5.tolist())
for ep in [0.05, 1e-4]:
    eb=ep/k; cl=R_cl(taup,lab,ep)
    a00=a_coeff(tau,y,eb,m,1e-5,1e-5)
    # backreaction slope b (per unit Delta), C->inf at fixed Delta
    from sw import bare_to_phys
    Cb=1e6; kb=bare_to_phys(Cb,tau,1.0)['kappa1']; b=Cb*(R_cl(kb*tau,kb*y,kb*eb)-R_cl(tau,y,eb))
    strict=(a00-b*m)/abs(R_cl(tau,y,eb))
    finite=k1*(R_cl(tau,y,eb)+a00/C-cl)/abs(cl)
    print(f'eps={ep}: a00={a00:.6f} b={b:.6f} quantum a00-b*m={a00-b*m:.6f}  c0(1-loop strict)={strict:.5f}  c0(1-loop at C={C:.3f})={finite:.5f}')
    out[f'oneloop_c0_eps{ep}']=dict(a00=a00,b=b,aq=a00-b*m,c0_strict=strict,c0_finiteC=finite)
json.dump(out,open('results_q4_fit.json','w'),indent=1,default=float)
