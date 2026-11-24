"""usage: python3 run2.py k1 h De ee labels what [winfile]"""
import numpy as np, sys, json, time
from core import *
k1=float(sys.argv[1]); h=float(sys.argv[2]); De=float(sys.argv[3]); ee=float(sys.argv[4])
labels=[float(x) for x in sys.argv[5].split(',')]
what=sys.argv[6].split(',')
winfile=sys.argv[7] if len(sys.argv)>7 else f'win_k{int(k1)}_De{De}_ee{ee}.json'
p=physical_map(k1,np.pi/4,2.)
W=json.load(open(winfile))
g={k:grid(lo-1,hi+1,h) for k,(lo,hi) in W.items()}
s=Setup(p['C'],p['ts'],De=De,ee=ee,grids=g)
print(f"# k1={k1} h={h} De={De} ee={ee} C={p['C']:.6f} kap={p['kap']:.6f} ts={p['ts']:.6f} sizes",{k:len(v[0]) for k,v in g.items()},flush=True)
for L in labels:
    t=time.time(); y=L/p['kap']
    r=all_gluings(s,y,what)
    cl=classical_reading(np.pi/4,L,p['kap']*s.eps); clb=classical_block(np.pi/4,L,p['kap']*s.eps,s.Delta,s.omega,s.hV)
    print(json.dumps(dict(k1=k1,h=h,De=De,ee=ee,label=L,cl=float(cl),cl_1mV=float(clb[1]),clb=float(clb[0]),
        **{k:[float(x) for x in readout(v,s.omega)] for k,v in r.items()},sec=round(time.time()-t,1))),flush=True)
