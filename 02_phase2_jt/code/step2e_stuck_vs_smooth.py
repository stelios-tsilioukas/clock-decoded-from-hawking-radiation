# Step 2E.3 exploratory scan (tossed-geodesic clock): QES stuck on C1 vs smooth out-QES. Output saved in results/step2e_stuck_vs_smooth.txt
import numpy as np, sys
sys.path.insert(0,'02_phase2_jt/code')
from step2_logjt import LogJT
from step2d_clock import Drop, gd
from step2e_qes_kink import KinkQES, lam_gyongyosi
from scipy.optimize import brentq
k=1e-4; M=LogJT(k); ydrop=0.5/k; ts=np.pi/4
kap=M.kappa(ydrop); phi_s=2*kap/np.sin(ts)
YS=ydrop-0.5
for _ in range(40):
    D=Drop(M,YS,phi_s); YS+=ydrop-D.y_radar
D=Drop(M,YS,phi_s); YH=D.Y_of_ah(D.point(D.lamH)[0])
def S_out_on_C1(lam,yB,Q):
    ah,bh=D.point(lam); Y=D.Y_of_ah(ah); return Q.Sgen(Y,bh,yB,'out'), Y, bh
def grad_out(lam,yB,Q):
    ah,bh=D.point(lam); Y=D.Y_of_ah(ah); return Q.grad(Y,bh,yB,'out'), Y, ah, bh
# test-clock crossing (mu=0): out-QES locus crosses C1 where tangential AND normal derivative vanish
Q0=KinkQES(M,D,0.0)
for lam_t in (0.0, 0.003, 0.01, 0.03):
    mu=4*kap*lam_t*np.tan(ts); Q=KinkQES(M,D,mu); lam_g,eps=lam_gyongyosi(D,M,mu) if mu>0 else (0,0)
    print(f'--- lambda={lam_g:.4f} mu={mu:.4f}')
    for yB in YH+np.log(4*kap/k)/(2*kap)+np.linspace(-1.5,1.5,13):
        # tangential derivative along C1: dS/dlam = gY*dY/dlam + gb*dbh/dlam
        def tang(lam):
            g,Y,ah,bh=grad_out(lam,yB,Q)
            dah=D.R*np.exp(-lam); dbh=D.R*np.exp(lam)
            dY=dah/(-ah*M.pieces(Y)['w'])          # dY/dlam = (dah/dlam)/(dah/dY)
            return g[0]*dY+g[1]*dbh
        lams=np.linspace(D.lamH-1.5,D.lamH+1.5,121); tv=[tang(l) for l in lams]
        idx=[i for i in range(120) if np.sign(tv[i])!=np.sign(tv[i+1])]
        out=[]
        for i in idx:
            l=brentq(tang,lams[i],lams[i+1],xtol=1e-13)
            g,Y,ah,bh=grad_out(l,yB,Q)
            # normal derivative of S_out along unit normal n = grad p (pointing out): d_n S = g^{ab} dS_a dp_b * 2
            p,pa,pb=Q.p_and_grad(ah,bh)
            # convert S-gradient to hat coords: d_ah = d_Y/(dah/dY), d_bh = gb
            Sa=g[0]/(-ah*M.pieces(Y)['w']); Sb=g[1]
            dnS=-(bh-ah)**2*(Sa*pb+Sb*pa)/2*2/2   # g^{ab}: -(b-a)^2/2 offdiag -> 2*(-(b-a)^2/2)*(Sa pb+Sb pa)/2
            dnS=-(bh-ah)**2*(Sa*pb+Sb*pa)/2
            theta=-dnS/mu if mu>0 else np.nan
            out.append((gd(l)-D.tauH, dnS, theta))
        print(f'  yB-YH={yB-YH:6.3f}: '+'; '.join(f'tau-tauH={t:+.5f} dnS={d:+.3e} theta={th:+.3f}' for t,d,th in out))

print("==== compare S_gen: stuck-on-C1 vs smooth out-QES (mu-independent as long as 0<=theta<=1)")
Q=KinkQES(M,D,0.03)
for yB in YH+np.log(4*kap/k)/(2*kap)+np.linspace(-1.0,4.0,11):
    def tang(lam):
        g,Y,ah,bh=grad_out(lam,yB,Q)
        dah=D.R*np.exp(-lam); dbh=D.R*np.exp(lam)
        return g[0]*dah/(-ah*M.pieces(Y)['w'])+g[1]*dbh
    lams=D.lamH+np.geomspace(1e-7,0.5,200)
    tv=[tang(l) for l in lams]; idx=[i for i in range(199) if np.sign(tv[i])!=np.sign(tv[i+1])]
    Yq,eta=M.qes(yB); bhq=eta*np.exp(M.lG(yB)-D.lGS); ahq=-np.exp(M.lG(Yq)-D.lGS)
    pq=Q.p_and_grad(ahq,bhq)[0]; Sq=Q.Sgen(Yq,bhq,yB,'out')
    s=f'yB-YH={yB-YH:6.3f}  smooth-out: p={pq:+.2e} S={Sq:.10f}'
    for i in idx:
        l=brentq(tang,lams[i],lams[i+1],xtol=1e-15); Sst,Y,bh=S_out_on_C1(l,yB,Q)
        s+=f' | stuck: tau-tauH={gd(l)-D.tauH:.3e} S-Ssmooth={Sst-Sq:+.3e}'
    print(s)
