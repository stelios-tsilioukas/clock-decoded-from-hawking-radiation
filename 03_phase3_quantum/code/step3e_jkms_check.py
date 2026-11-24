"""Step 3e: the matter three-point amplitude from the Jafferis-Kolchmeyer-Mukhametzhanov-Sonner disk rules
(2209.02131, eqs. 2.5-2.10): three generalized-free-field chords with their shared endpoints split by a boundary
segment eps (six insertions, no crossings -> no 6j symbol). The interior region then has three boundary segments of
length eps, the caps lose eps/2 each side.  eps -> 0 must reproduce Exact.log_triangle (the four-region integral)."""
import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3e_matter_clock as M
from step3d_backreaction import lmu

def log_triangle_split(X, ts, dy, wa, wb, wc, eps, m_grid):
    s = X.s; k, w = X.grid(m_grid, ts)
    t0 = X.bM - 2*s*ts - s*eps; ta = s*(ts + 1j*dy - eps); tb = s*(ts - 1j*dy - eps)   # each cap loses eps (two half-gaps)
    offC, Cv = X._inner(wc, t0, k, w); offa, Av = X._inner(wa, ta, k, w); offb, Bv = X._inner(wb, tb, k, w)
    E = lmu(k) + offC + offa + offb - k**2*(3*s*eps); o = E.max()
    return o + np.log((np.exp(E - o)*Cv*Av*Bv*w).sum()) - X.logZ() + 2*(wa + wb + wc)*np.log(s)

if __name__ == "__main__":
    ts = np.pi/4
    for C in (10.0, 2.0):
        X = M.Exact(C, N=4000)
        for (D, om, h) in ((2.0, 0.4, 1.0), (0.6, 0.2, 0.5)):
            wa, wb, wc = M.weights(D, om, h)
            for dy in (-0.347, 0.0):
                ref = X.log_triangle(ts, [dy], wa, wb, wc, m_grid=D + om/2)[0]
                d = [log_triangle_split(X, ts, dy, wa, wb, wc, e, D + om/2) - ref for e in (1e-1, 1e-2, 1e-3, 1e-4)]
                print(f"C={C} Delta={D} omega={om} h={h} dy={dy:+.3f}: split - four-region at eps=1e-1..1e-4: "
                      + "  ".join(f"{z.real:+.2e}{z.imag:+.2e}i" for z in d), flush=True)
