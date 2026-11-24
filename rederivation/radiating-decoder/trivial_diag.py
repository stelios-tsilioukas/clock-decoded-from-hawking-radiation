"""Diagnostic of the Delta_e -> 0 limit: split the emission kernel g_De(s,E) into a core |s-E|<c*De and a tail."""
import numpy as np, json, sys, time
from core import *
k1=20; p=physical_map(k1,np.pi/4,2.)
W=json.load(open('win_k20_De1.0_ee0.0125.json'))
L=float(sys.argv[1]); De=float(sys.argv[2]); cs=[float(x) for x in sys.argv[3].split(',')]
hf=De/6
g={k:grid(lo-1,hi+1,0.125) for k,(lo,hi) in W.items()}
g['S']=grid(W['B'][0]-1,W['B'][1]+1,hf); g['E']=grid(W['E'][0]-1,W['E'][1]+1,hf)
s=Setup(p['C'],p['ts'],De=De,grids=g); y=L/p['kap']; sb=s.sb; ee=s.ee
D,hD=g['D']; S,hS=g['S']; E,hE=g['E']
lD=logmu(D)-s.tD*D**2+np.log(hD)
lFS=lse(lD[:,None]+logg(s.Dchi,D[:,None],S[None,:]),0)
LSa=[]
for a in (0,1):
    LIS=np.concatenate([s.logItri(a,y,D,S[i:i+1000]) for i in range(0,len(S),1000)],axis=1)
    LSa.append(lse(lD[:,None]+LIS,0)); del LIS
lws=logmu(S)+np.log(hS)-sb*(s.ts+s.eps/2-2*ee-1j*y)*S**2
lwP=logmu(S)+np.log(hS)-sb*(2*s.ts-2*ee)*S**2+lFS
# Y0 on the S grid for reference
lY=logmu(S)+np.log(hS)-lFS-sb*s.eps*S**2
rY={(a,b):lse(lY+LSa[a]+np.conj(LSa[b]),0) for a in (0,1) for b in (0,1)}
print('Y0 on fine S grid',readout(rY,0.2),flush=True)
for c in cs:
  for mode in ('core','full'):
    lP=np.empty(len(E)); LK=[np.empty(len(E),complex) for a in (0,1)]
    for j in range(0,len(E),1500):
        lge=logg(De,S[:,None],E[None,j:j+1500])
        if mode=='core': lge=np.where(abs(S[:,None]-E[None,j:j+1500])<c*De,lge,-np.inf)
        lP[j:j+1500]=lse(lwP[:,None]+lge,0).real
        for a in (0,1): LK[a][j:j+1500]=lse((lws+LSa[a])[:,None]+lge,0)
    ok=np.isfinite(lP)
    lE=logmu(E)+np.log(hE)-2*sb*ee*E**2-lP
    rho={(a,b):lse((lE+LK[a]+np.conj(LK[b]))[ok],0) for a in (0,1) for b in (0,1)}
    print(json.dumps(dict(De=De,c=c,mode=mode,dece=[float(x) for x in readout(rho,0.2)])),flush=True)
    if mode=='full': break
