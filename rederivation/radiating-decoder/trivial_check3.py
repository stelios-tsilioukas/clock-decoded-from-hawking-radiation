"""Delta_e -> 0 check of the decorated decoded formula with memory-light chunking over E.
Fine grids (h_f = Delta_e/6) on the sigma (S) and E regions, h=0.125 elsewhere."""
import numpy as np, json, sys, time
from core import *
k1=20; p=physical_map(k1,np.pi/4,2.)
W=json.load(open('win_k20_De1.0_ee0.025.json')); W['B']=json.load(open('win_k20_De1.0_ee0.0125.json'))['B']
L=float(sys.argv[1]); EE=float(sys.argv[3]); Des=[float(x) for x in sys.argv[2].split(',')]
for De in Des:
    hf=min(De/6,0.125)
    g={k:grid(lo-1,hi+1,0.125) for k,(lo,hi) in W.items()}
    g['S']=grid(W['B'][0]-1,W['B'][1]+1,hf)  # sigma window = full s-window of Y0 (needed when Delta_e->0)
    g['E']=grid(W['E'][0]-1,W['E'][1]+1,hf)
    s=Setup(p['C'],p['ts'],De=De,ee=EE,grids=g); y=L/p['kap']; sb=s.sb; ee=s.ee
    t=time.time()
    D,hD=g['D']; S,hS=g['S']; E,hE=g['E']
    lD=logmu(D)-s.tD*D**2+np.log(hD)
    lFS=lse(lD[:,None]+logg(s.Dchi,D[:,None],S[None,:]),0)
    LSa=[]
    for a in (0,1):
        LIS=np.concatenate([s.logItri(a,y,D,S[i:i+1000]) for i in range(0,len(S),1000)],axis=1)
        LSa.append(lse(lD[:,None]+LIS,0)); del LIS
    lws=logmu(S)+np.log(hS)-sb*(s.ts+s.eps/2-2*ee-1j*y)*S**2
    lwP=logmu(S)+np.log(hS)-sb*(2*s.ts-2*ee)*S**2+lFS
    lP=np.empty(len(E)); LK=[np.empty(len(E),complex) for a in (0,1)]
    for j in range(0,len(E),1500):
        lge=logg(De,S[:,None],E[None,j:j+1500])
        lP[j:j+1500]=lse(lwP[:,None]+lge,0).real
        for a in (0,1): LK[a][j:j+1500]=lse((lws+LSa[a])[:,None]+lge,0)
    lE=logmu(E)+np.log(hE)-2*sb*ee*E**2-lP
    rho={(a,b):lse(lE+LK[a]+np.conj(LK[b]),0) for a in (0,1) for b in (0,1)}
    print(json.dumps(dict(De=De,ee=EE,hf=hf,label=L,dece=[float(x) for x in readout(rho,0.2)],nS=len(S),nE=len(E),sec=round(time.time()-t,1))),flush=True)
