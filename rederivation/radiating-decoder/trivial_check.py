"""Numerical check that the decorated decoded formula -> Y^0 and the one-sided with emission -> one-sided
without emission when the emission kernel becomes trivial (Delta_e -> 0, g_{Delta_e}(s,E) -> delta(s-E)/rho(E))."""
import numpy as np, json, sys, time
from core import *
k1=20; p=physical_map(k1,np.pi/4,2.)
W=json.load(open('win_k20_De1.0_ee0.0125.json'))
L=float(sys.argv[1]) if len(sys.argv)>1 else -1.0
for De,hf in [(1.0,0.125),(0.5,0.0625),(0.25,0.03125)]:
    g={k:grid(lo-1,hi+1,0.125) for k,(lo,hi) in W.items()}
    g['S']=grid(W['S'][0]-1,W['S'][1]+1,hf); g['E']=grid(W['E'][0]-1,W['E'][1]+1,hf)
    s=Setup(p['C'],p['ts'],De=De,grids=g)
    t=time.time(); r=all_gluings(s,L/p['kap'],('os0','ose','y0','dece'))
    print(json.dumps(dict(De=De,hf=hf,label=L,**{k:[float(x) for x in readout(v,0.2)] for k,v in r.items()},sec=round(time.time()-t,1))),flush=True)
