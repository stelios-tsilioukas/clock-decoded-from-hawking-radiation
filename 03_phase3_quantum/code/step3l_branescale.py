import numpy as np, sys
sys.path.insert(0,'rederivation/rmt-absorption'); sys.path.insert(0,'03_phase3_quantum/code')
from planar import Planar
import step3i_rmt as R
y, eps = 0.9, 0.05
def ours_rel(M, w):
    p = M.p; Ls, E, L = M.Ls, M.Eb, M.L; f2 = np.asarray(p['f2']); br = Ls*np.exp(-p['tb']*E)*f2
    y0 = Ls*np.exp(-p['tc']*E)*(M.s0 @ br)/L; y1 = Ls*np.exp(-p['tc']*E)*(M.s1 @ br)/L
    do = np.sqrt((y0/y0.sum())/(y1/y1.sum())); wag = w.sum(axis=1)
    return (wag.sum(axis=1)*do).sum()/w.sum()
for nm, base in (('thread2', R.REGIMES['thread2']), ('thread', dict(R.REGIMES['thread'], a=0.4))):
    print(f"\n{nm}: scale brane-side block dimensions (blocks 1-4) by lambda")
    print(f"{'lambda':>7} {'thread/cap':>11} | {'agent':>20} {'ours':>20} | {'|q|-1':>9} {'arg q (deg)':>11} | {'(|q|-1)*ratio':>13}")
    Ls0 = list(base['Ls'])
    for lam in (1, 3, 10, 30, 100, 300):
        p = dict(base); p['Ls'] = [Ls0[0]] + [int(round(v*lam)) for v in Ls0[1:]]
        M = R.RMT(p); P = Planar(p, np.inf); r = P.predict(y, eps); w = P.wtensor(y)
        a = (w*r['D'][:, None, :]).sum()/w.sum(); o = ours_rel(M, w); q = a/o; tcr = M.thread_cap_ratio()
        print(f"{lam:7d} {tcr:11.4g} | {a.real:+.5f}{a.imag:+.5f}i {o.real:+.5f}{o.imag:+.5f}i | {abs(q)-1:+9.5f} {np.degrees(np.angle(q)):+11.4f} | {(abs(q)-1)*tcr:+13.4f}", flush=True)
