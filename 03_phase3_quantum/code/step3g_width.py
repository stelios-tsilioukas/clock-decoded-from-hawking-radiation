"""Width of the canonical Page transition for decoding: spread of log rho-bar(k) over the physical energy distribution,
compared with S - S0, and the c1(x) curve (independent of which clock-related region is filtered: any boundary energy
of the black hole with the clock has this distribution up to O(m/C))."""
import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3g_decoding as D
for C in (40.0, 10.0, 2.0):
    X = D.Decoder(C, np.pi/4, 2.0, S0=8.0, N=2500)
    k = X.k; lrho = np.log(X.d) + X.S0 + D.lmu(k) + np.log(X.wq); ly = X.log_yc()
    lp = lrho + ly; lp -= np.logaddexp.reduce(lp); p = np.exp(lp)
    Sloc = D.lmu(k) + X.S0 + np.log(X.d)              # log density of states at k (per unit k)
    mS = (p*Sloc).sum(); sS = np.sqrt((p*(Sloc - mS)**2).sum())
    _, _, Sbh, _ = X.filter(0.0)
    xs = np.arange(-30, 31, 2.0); c1 = [X.filter(Sbh + x)[1] for x in xs]
    def cross(level):
        return np.interp(level, c1, xs)
    print(f"C={C:5.1f} (S-S0 = 2 pi C = {2*np.pi*C:6.1f}): sigma of local entropy over the ensemble = {sS:5.2f} nats;"
          f" sqrt(S-S0) = {np.sqrt(2*np.pi*C):5.2f};  c1 = 0.1, 0.5, 0.9, 0.99 at x = {cross(0.1):+.1f}, {cross(0.5):+.1f}, {cross(0.9):+.1f}, {cross(0.99):+.1f}")
