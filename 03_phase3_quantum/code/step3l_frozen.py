import numpy as np, sys
sys.path.insert(0,'rederivation/rmt-absorption'); sys.path.insert(0,'03_phase3_quantum/code')
from planar import Planar
import step3i_rmt as R
y, eps = 0.9, 0.05
class Frozen(Planar):
    """brane-side resolvent frozen: x_gamma = 1 (no sharing of brane-side indices between replicas)"""
    def xfun(self, pg, w): return np.ones_like(pg)
def ours_rel(M, w):
    p = M.p; Ls, E, L = M.Ls, M.Eb, M.L; f2 = np.asarray(p['f2']); br = Ls*np.exp(-p['tb']*E)*f2
    y0 = Ls*np.exp(-p['tc']*E)*(M.s0 @ br)/L; y1 = Ls*np.exp(-p['tc']*E)*(M.s1 @ br)/L
    do = np.sqrt((y0/y0.sum())/(y1/y1.sum())); wag = w.sum(axis=1)
    return (wag.sum(axis=1)*do).sum()/w.sum()
base, base2 = dict(R.REGIMES['thread'], a=0.4), R.REGIMES['thread2']
cases = [('thread', base), ('thread tb=1', dict(base, tb=1.0)), ('thread2', base2), ('thread2 lam=30', dict(base2, Ls=[16]+[v*30 for v in base2['Ls'][1:]])),
         ('mixed', dict(R.REGIMES['mixed'], a=0.4)), ('cap', dict(R.REGIMES['cap'], a=0.4))]
print(f"{'case':>15} | {'exact planar':>20} {'frozen x=1':>20} {'ours (threading)':>20} | {'frozen/ours-1':>13} | <p_gamma> (k=inf, s-avg)")
for nm, p in cases:
    M = R.RMT(p); P = Planar(p, np.inf); F = Frozen(p, np.inf); w = P.wtensor(y)
    rE = P.predict(y, eps); rF = F.predict(y, eps)
    e = (w*rE['D'][:, None, :]).sum()/w.sum(); f = (w*rF['D'][:, None, :]).sum()/w.sum(); o = ours_rel(M, w)
    pg = 1/rE['I']['T'].mean() if False else None
    # typical brane self-energy: p_gamma at s where the integrand of m peaks (s ~ Lambda)
    G, X = P.G, P.X; pgam = 1/X - 1                                        # x = 1/(1+p)
    sp = np.exp(P.vs); wts = sp**0.5*(1 - sp[:, None]*G[:, :5]).mean(1); i = np.argmax(wts)
    print(f"{nm:>15} | {e.real:+.5f}{e.imag:+.5f}i {f.real:+.5f}{f.imag:+.5f}i {o.real:+.5f}{o.imag:+.5f}i | {abs(f/o)-1:+.2e} | {np.round(pgam[i],3)}")
print("\nother-placement / own (unheralded background), exact vs frozen brane resolvent:")
for nm, p in cases[:4]:
    P = Planar(p, np.inf); F = Frozen(p, np.inf); rE = P.predict(y, eps); rF = F.predict(y, eps)
    print(f"{nm:>15}: exact {rE['other']/rE['own']:+.4f}   frozen {rF['other']/rF['own']:+.4f}")
