import numpy as np, sys, json, time
from run import setup, compute
from core import *
k1=float(sys.argv[1]); labels=[float(x) for x in sys.argv[2].split(',')]
for h,pad in [(0.2,1.0),(0.125,1.0),(0.1,1.0),(0.125,8.0)]:
    s,p=setup(k1,h,pad=pad)
    for L in labels:
        t=time.time(); r=compute(s,p,L,['os0','ose','y0','dece'])
        print(json.dumps(dict(h=h,pad=pad,label=L,**{k:[float(v[0]),float(v[1])] for k,v in r.items()},sec=round(time.time()-t,1))),flush=True)
