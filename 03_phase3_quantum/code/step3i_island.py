"""
Step 3i: the frozen stopwatch for a clock that is in the ISLAND on the time-symmetric slice (tau_s = 3 pi/4),
decoded with the exact planar resolvent.  Here the threading pairing (the one Step 3h kept) dominates the cap
pairing by tan(tau_s/2)^{4 Delta} = e^{+7.05} at Delta = 2 (step3i_thread_vs_cap.py), so the Step 3h decoding formula
is the leading planar term; caps correct it at relative order e^{-7}.

Clock: Delta = 2, omega = 0.2 (levels psi_0, psi_1), stop signal h = 1 at real time y (pulse width eps = 0.05),
degenerate frozen levels chi_0, chi_1 of dimension Delta + omega/2.  At tau_s = 3pi/4 the clock is at rest behind the
horizon at t = 0 and every absorption event lies in the future interior; labels dy = y x1 in {-2, -1, 0, +1}
(dy = 0: absorption on the phi = 0 surface; classical readings tau_cl - tau_cl(0) = -0.611, -0.378, 0, +0.378).
Outputs: decoded (exact resolvent; canonical for comparison) and physical one-sided readings and visibilities across
x = log k - S_BH, and the two-time intervals between stop labels (convention-free, see step3i_interval.py).
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_stopwatch as SW, step3d_backreaction as B
from step3e_matter_clock import tau_cl
from step3h_replica import lse
from step3i_clock_exact import exact_filter

OUT = '03_phase3_quantum/results/'
XS = [-12, -8, -4, 0, 4, 8, 12, 20, 30, np.inf]
DYS = [-2.0, -1.0, 0.0, 1.0]

def run(C, ts=3*np.pi/4, D=2.0, h=1.0, eps=0.05, om=0.2, xs=XS):
    x1 = B.saddle(D/C, ts)[1]; labs = {f"dy={dy:+.0f}": dy/x1 for dy in DYS}
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2600, max(1400, kmax/0.04)))
    t0 = time.time(); S = SW.Stopwatch(C, ts, D, om, h, eps, kmax, N)
    el = {nm: S.elements(y) for nm, y in labs.items()}
    print(f"C={C} tau_s={ts/np.pi:.2f}pi: elements in {time.time()-t0:.0f}s (N={N}, x1={x1:.4f})", flush=True)
    filt = {x: (S.filter(x), exact_filter(S, x)[0]) for x in xs}
    rows = {}
    for nm, (phys, Y) in el.items():
        Vp = np.exp(phys[(0, 1)] - 0.5*(phys[(0, 0)] + phys[(1, 1)]))
        r = dict(y=labs[nm], classical=float(tau_cl(ts, labs[nm]*x1) - tau_cl(ts, 0.0)), phys=[float(abs(Vp)), float(np.angle(Vp)/om)], dec_exact=[], dec_canonical=[])
        for x in xs:
            for key, lG in (('dec_canonical', filt[x][0]), ('dec_exact', filt[x][1])):
                rr = {k: lse(v + lG) for k, v in Y.items()}
                Vd = np.exp(rr[(0, 1)] - 0.5*(rr[(0, 0)] + rr[(1, 1)])); r[key].append([float(abs(Vd)), float(np.angle(Vd)/om)])
        rows[nm] = r
        for i, x in enumerate(xs):
            e, c = r['dec_exact'][i], r['dec_canonical'][i]
            print(f"C={C} {nm}: x={x:+6.1f} reading phys {r['phys'][1]:+.5f} dec exact {e[1]:+.5f} (canonical {c[1]:+.5f}; exact - phys {e[1]-r['phys'][1]:+.5f}) "
                  f"| |V| phys {r['phys'][0]:.5f} dec {e[0]:.5f}  [classical {r['classical']:+.4f}]", flush=True)
    c1 = {str(x): float(np.exp(lse(S.lP + filt[x][1]))) for x in xs}
    print(f"C={C} c1 exact: " + "  ".join(f"x={x}: {c1[str(x)]:.4f}" for x in xs), flush=True)
    return dict(C=C, tau_s=ts, N=N, x1=x1, labels=labs, xs=[x if np.isfinite(x) else 'inf' for x in xs], rows=rows, c1=c1)

if __name__ == "__main__":
    Cs = [float(c) for c in sys.argv[1].split(',')] if len(sys.argv) > 1 else [10.0, 2.0]
    res = [run(C) for C in Cs]
    tag = '_'.join(f"{int(c)}" for c in Cs)
    json.dump(res, open(OUT + f'step3i_island_C{tag}.json', 'w'), indent=1, default=float)
