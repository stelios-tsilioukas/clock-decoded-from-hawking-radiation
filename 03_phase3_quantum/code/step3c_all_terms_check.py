import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'03_phase3_quantum/code')
import step3c_all_terms as S
from step3c_all_terms import AllTerms, KF, Kh, gfun
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
# independent evaluation of the finite part: K = -(1/4pi^2) Int dd (h(d) - h(0))/d^2, by adaptive quadrature in k-space
X=AllTerms(2.0,np.pi/4,2.0)
# term (a,s) at Dx=0, rebuilt here
sc,ts,m,p=X.sc,X.ts,X.m,X.p; t1=sc*ts
E=X.lmu_k[None,:]+X.lg_sk-(X.k**2*t1)[None,:]; B1=(np.exp(E)*X.w[None,:]).sum(axis=1)
fBB=X.even_spline((B1*B1).real)
H=lambda a,s: np.exp(-X.t0*a**2)*gfun(m-p/2,a,s)*fBB(s)
sig=np.linspace(-10,10,1601); wsig=np.full(1601,sig[1]-sig[0]); wsig[0]*=.5; wsig[-1]*=.5
hfun=lambda d: float((H(sig+d/2,sig-d/2)*wsig).sum().real)
h0=hfun(0.0)
direct=-(1/(4*np.pi**2))*(quad(lambda d:(hfun(d)+hfun(-d)-2*h0)/d**2,1e-6,20,limit=400,epsabs=0,epsrel=1e-9)[0] - 2*h0/20)
print('term (a,s): FFT route',X.K2(H).real,' direct finite-part route',direct)
