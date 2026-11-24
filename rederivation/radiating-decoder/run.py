"""Production: readings for list of physical labels.
usage: python3 run.py k1 h De ee labels(comma) [what=os0,ose,y0,dece,hs,hd] [winscale]"""
import numpy as np, sys, json, time
from core import *

def setup(k1, h, De=1.0, ee=0.0125, winfile=None, pad=1.0, tsp=np.pi/4, m=2., Delta=2.):
    p = physical_map(k1, tsp, m)
    if winfile is None:
        winfile = f'win_k{int(k1)}_De1.0_ee0.0125.json'
    W = json.load(open(winfile))
    g = {}
    for k, (lo, hi) in W.items():
        g[k] = grid(lo - pad, hi + pad, h)
    s = Setup(p['C'], p['ts'], Delta=Delta, De=De, ee=ee, grids=g)
    return s, p

def compute(s, p, label, what):
    y = label / p['kap']
    out = {}
    if 'os0' in what: out['os0'] = readout(s.onesided(y, emission=False), s.omega)
    if 'ose' in what: out['ose'] = readout(s.onesided(y, emission=True), s.omega)
    if 'y0' in what: out['y0'] = readout(s.decoded0(y), s.omega)
    if 'hs' in what: out['hs'] = readout(s.decoded0(y, Bweight=lambda S: s.logWe(S, True)), s.omega)
    if 'dece' in what: out['dece'] = readout(s.decoded_e(y), s.omega)
    if 'hd' in what: out['hd'] = readout(s.decoded_e(y, share_D=True), s.omega)
    return out

if __name__ == '__main__':
    k1 = float(sys.argv[1]); h = float(sys.argv[2]); De = float(sys.argv[3]); ee = float(sys.argv[4])
    labels = [float(x) for x in sys.argv[5].split(',')]
    what = sys.argv[6].split(',') if len(sys.argv) > 6 else ['os0', 'ose', 'y0', 'dece']
    s, p = setup(k1, h, De, ee)
    print(f"# k1={k1} h={h} De={De} ee={ee} C={p['C']:.6f} kap={p['kap']:.6f} ts={p['ts']:.6f}", flush=True)
    for L in labels:
        t = time.time()
        r = compute(s, p, L, what)
        cl = classical_reading(np.pi/4, L, p['kap']*s.eps)
        clb = classical_block(np.pi/4, L, p['kap']*s.eps, s.Delta, s.omega, s.hV)
        line = dict(k1=k1, h=h, De=De, ee=ee, label=L, cl=cl, cl_1mV=clb[1],
                    **{k: [float(v[0]), float(v[1])] for k, v in r.items()}, sec=round(time.time()-t, 1))
        print(json.dumps(line), flush=True)
