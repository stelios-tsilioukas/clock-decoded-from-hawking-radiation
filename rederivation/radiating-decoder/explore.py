import numpy as np, sys
from core import *
k1=float(sys.argv[1]) if len(sys.argv)>1 else 20
p=physical_map(k1,np.pi/4,2.)
h=0.5
g=dict(A=grid(0,250,h),I=grid(0,250,h),D=grid(0,120,h),B=grid(0,400,h),E=grid(0,500,h),S=grid(0,300,h))
s=Setup(p['C'],p['ts'],grids=g)
def marg(name,x,l):
    l=l.real-l.real.max(); idx=np.where(l>-40)[0]
    print(f"{name}: peak {x[np.argmax(l)]:.1f} window(e^-40) [{x[idx[0]]:.1f},{x[idx[-1]]:.1f}]  edges {l[0]:.1f} {l[-1]:.1f}")
D,_=g['D'];B,_=g['B'];I,_=g['I'];A,_=g['A']
y=0.0
# one-sided support
for emis in (False,True):
    lD=logmu(D)-s.tD*D**2; lB=logmu(B)+s.logWe(B,emis)
    LI=s.logItri(0,y,D,B)
    T=lD[:,None]+lB[None,:]-logg(s.Dchi,D[:,None],B[None,:])+2*LI.real
    print('onesided emission',emis); marg('D',D,lse(T,1)); marg('B',B,lse(T,0))
# I marginal given D*,B*: 
LIc=s.logcap(0,y,I)+logmu(I)
w=s.weights(0)
lD=logmu(D)-s.tD*D**2; lB=logmu(B)+s.logWe(B,False)
T=lD[:,None]+lB[None,:]-logg(s.Dchi,D[:,None],B[None,:])+2*s.logItri(0,y,D,B).real
iD,iB=np.unravel_index(np.argmax(T),T.shape)
lI=LIc+logg(w['psichi'],D[iD],I)+logg(w['Vchi'],I,B[iB]); marg('I (at D*,B*)',I,lI)
iI=np.argmax(lI.real)
lA=logmu(A)-s.sb*(s.ts-s.eps/2)*A**2+logg(w['psiV'],A,I[iI]); marg('A (at I*)',A,lA)
# decoded support (no emission)
S=B
lS=logmu(S)-s.logF_chi(S)-s.sb*s.eps*S**2
LS=s.logS(0,y,S); marg('s (Y0)',S,lS+2*LS.real)
# D marginal inside S_a at s*
iS=np.argmax((lS+2*LS.real)); LI=s.logItri(0,y,D,S[iS:iS+1]); marg('D in S_a(s*)',D,lD+LI[:,0])
# decoded emission
S,_=g['S'];E,_=g['E']
lge=logg(s.De,S[:,None],E[None,:])
lFc=s.logF_chi(S)
lPint=(logmu(S)-s.sb*(2*s.ts-2*s.ee)*S**2+lFc)[:,None]+lge
lP=lse(lPint,0)
lE=logmu(E)-2*s.sb*s.ee*E**2-lP
lSa=s.logS(0,y,S)
lws=logmu(S)-s.sb*(s.ts+s.eps/2-2*s.ee)*S**2
LK=lse((lws+lSa)[:,None]+lge,0)
marg('E (dec_e)',E,lE+2*LK.real)
iE=np.argmax(lE+2*LK.real)
marg('sigma (dec_e, at E*)',S,lws+lSa.real+lge[:,iE])
marg('sigma in P (at E*)',S,lPint[:,iE])
iSs=np.argmax(lws+lSa.real+lge[:,iE]); LI=s.logItri(0,y,D,S[iSs:iSs+1]); marg('D in S_a(sigma*)',D,lD+LI[:,0])
