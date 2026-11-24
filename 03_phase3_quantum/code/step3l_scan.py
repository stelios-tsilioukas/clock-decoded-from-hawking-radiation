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
print(f"{'regime':>28} {'thread/cap':>11} | {'agent own/ovl*':>20} {'ours':>20} | {'|ratio|-1':>9} {'arg ratio (deg)':>15}")
base = R.REGIMES['thread']; base2 = R.REGIMES['thread2']
cases = [('thread (tc=8,tb=.05)', base)] + [(f'thread tc={tc}', dict(base, tc=tc)) for tc in (4.0, 12.0, 16.0, 24.0)] \
      + [(f'thread tb={tb}', dict(base, tb=tb)) for tb in (0.0, 0.5, 1.0, 2.0)] \
      + [('thread2 (tc=8,tb=0)', base2)] + [(f'thread2 tc={tc}', dict(base2, tc=tc)) for tc in (4.0, 12.0, 16.0)] + [(f'thread2 tb={tb}', dict(base2, tb=tb)) for tb in (0.5, 1.0, 2.0)] \
      + [('mixed', R.REGIMES['mixed']), ('cap', R.REGIMES['cap'])]
for nm, p in cases:
    p = dict(p); p.setdefault('a', 0.4)
    M = R.RMT(p); P = Planar(p, np.inf); r = P.predict(y, eps); w = P.wtensor(y)
    a = (w*r['D'][:, None, :]).sum()/w.sum(); o = ours_rel(M, w); q = a/o
    print(f"{nm:>28} {M.thread_cap_ratio():11.3g} | {a.real:+.5f}{a.imag:+.5f}i {o.real:+.5f}{o.imag:+.5f}i | {abs(q)-1:+9.5f} {np.degrees(np.angle(q)):+15.3f}", flush=True)
