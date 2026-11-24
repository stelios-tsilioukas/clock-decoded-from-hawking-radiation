import numpy as np, sys
sys.path.insert(0,'rederivation/rmt-absorption'); sys.path.insert(0,'03_phase3_quantum/code')
from planar import Planar
from step3i_rmt import RMT, REGIMES
np.set_printoptions(precision=4, suppress=True, linewidth=160)
y, eps = 0.9, 0.05
for reg in ('thread','thread2'):
    for k in (np.inf, 20000):
        P = Planar(reg, k); r = P.predict(y, eps); D = r['D']; w = P.wtensor(y)
        M = RMT(REGIMES[reg]); p = M.p; Ls, E, L = M.Ls, M.Eb, M.L; f2 = np.asarray(p['f2'])
        br = Ls*np.exp(-p['tb']*E)*f2
        y0 = Ls*np.exp(-p['tc']*E)*(M.s0 @ br)/L; y1 = Ls*np.exp(-p['tc']*E)*(M.s1 @ br)/L
        p0, p1 = y0/y0.sum(), y1/y1.sum(); do = np.sqrt(p0/p1)
        wag = w.sum(axis=1)                              # weight over (alpha=V row, gamma=brane)
        own_rel = (w*D[:,None,:]).sum()/w.sum()          # agent own / overlap*
        ours_rel = (wag.sum(axis=1)*do).sum()/w.sum()    # ours / overlap*
        Dbar = (wag*D).sum(axis=1)/wag.sum(axis=1)       # gamma-averaged agent factor per alpha
        print(f"\n== {reg} k={k}: own/overlap* agent {own_rel:.5f}  ours(threading) {ours_rel:.5f}  ratio {own_rel/ours_rel:.5f}")
        print("   alpha weight share     ", np.abs(wag.sum(1))/np.abs(wag.sum()))
        print("   ours d(alpha)=sqrt(p0/p1)", do)
        print("   agent Dbar(alpha)        ", Dbar)
        print("   agent D(alpha,gamma)/Dbar spread (max|.-1| over gamma with weight>1e-3):",
              [float(np.max(np.abs(D[a,np.abs(wag[a])>1e-3*np.abs(wag[a]).max()]/Dbar[a]-1))) if np.abs(wag[a]).sum()>0 else 0 for a in range(len(Ls))])
