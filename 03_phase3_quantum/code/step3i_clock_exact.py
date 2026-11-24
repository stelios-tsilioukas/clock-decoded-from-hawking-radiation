"""
NOTE: this runs the Step 3h (threading-only) formula at tau_s = pi/4, where caps actually dominate; it is kept as the
exact-vs-canonical resolvent comparison only. Physical decoded results for an island clock: step3i_island.py.

Step 3i: decoded clock quantities with the EXACT planar resolvent (replaces PSSY's canonical approximation
(lambda_0 + w)^{-1/2} -> J_a(s) from step3i_resolvent.ExactResolvent).

(A) frozen stopwatch (degenerate chi register), C = 10 and 2: decoded visibility and reading across the Page
    transition, at four stop labels (the same decoder for all labels -> intervals, see step3i_interval.py);
(B) non-degenerate register: decoded density-matrix positivity 1 - |rho10|^2/(rho00 rho11) (Step 3h section 3b).
Filter: canonical  G = w/(lambda_0 + w);  exact  G = w J^2.
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_stopwatch as SW, step3h_density as DN, step3d_backreaction as B
from step3h_replica import lse
from step3i_resolvent import ExactResolvent

OUT = '03_phase3_quantum/results/'
XS = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 20, 30, np.inf]

def exact_filter(S, x, M=4096):
    """log G_exact(s) = log w + 2 log J for the degenerate code (both levels share lP)."""
    if not np.isfinite(x): return np.zeros(S.lrho.size), (1.0, 1.0)
    R = ExactResolvent(S.lrho, [S.lP, S.lP], S.Sbh + x, M=M)
    lJ = R.logJ()[0]; return S.lwc + 2*lJ, R.sum_rules()

def stopwatch(C, ts=np.pi/4, D=2.0, h=1.0, eps=0.05, om=0.2):
    x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    labs = {'y_H-1': yH - 1/x1, 'horizon': yH, 'phi=0': 0.0, '0.9 inner': -0.9*yH}
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2600, max(1400, kmax/0.04)))
    S = SW.Stopwatch(C, ts, D, om, h, eps, kmax, N)
    t0 = time.time(); el = {nm: S.elements(y) for nm, y in labs.items()}
    print(f"C={C}: elements for 4 labels in {time.time()-t0:.0f}s (N={N})", flush=True)
    filt = {}
    for x in XS:
        lGc = S.filter(x); lGe, sr = exact_filter(S, x)
        filt[x] = (lGc, lGe, sr)
        # robustness: M doubled at x = 0
        if x == 0:
            lGe2, _ = exact_filter(S, x, M=8192); filt['M8192'] = float(np.max(np.abs(np.exp(lGe2) - np.exp(lGe))))
    rows = {}
    for nm, (phys, Y) in el.items():
        Vp = np.exp(phys[(0, 1)] - 0.5*(phys[(0, 0)] + phys[(1, 1)]))
        r = dict(phys=[float(abs(Vp)), float(np.angle(Vp)/om)], dec_canonical=[], dec_exact=[])
        for x in XS:
            lGc, lGe, sr = filt[x]
            for key, lG in (('dec_canonical', lGc), ('dec_exact', lGe)):
                rr = {k: lse(v + lG) for k, v in Y.items()}
                Vd = np.exp(rr[(0, 1)] - 0.5*(rr[(0, 0)] + rr[(1, 1)]))
                r[key].append([float(abs(Vd)), float(np.angle(Vd)/om)])
        rows[nm] = r
        for i, x in enumerate(XS):
            c, e = r['dec_canonical'][i], r['dec_exact'][i]
            print(f"C={C} {nm:9s} x={x:+6.1f}: reading phys {r['phys'][1]:+.5f} | dec canonical {c[1]:+.5f} exact {e[1]:+.5f} "
                  f"(exact-phys {e[1]-r['phys'][1]:+.5f}) | |V| phys {r['phys'][0]:.5f} canon {c[0]:.5f} exact {e[0]:.5f}", flush=True)
    c1 = {}
    for x in XS:
        lGc, lGe, sr = filt[x]
        c1[str(x)] = dict(canonical=float(np.exp(lse(S.lP + lGc))), exact=float(np.exp(lse(S.lP + lGe))), sum_rules=list(map(float, sr)))
    print(f"C={C} c1 (exact | canonical): " + "  ".join(f"x={x}: {c1[str(x)]['exact']:.4f}|{c1[str(x)]['canonical']:.4f}" for x in XS), flush=True)
    print(f"C={C} max |G(M=8192) - G(M=4096)| at x=0: {filt['M8192']:.2e}", flush=True)
    return dict(C=C, N=N, labels=labs, xs=[x if np.isfinite(x) else 'inf' for x in XS], rows=rows, c1=c1, M_check=filt['M8192'])

def density_exact(C, om, ts=np.pi/4, eps=0.05, S0=8.0, dcode=2):
    x1 = B.saddle(2.0/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2600, max(1400, kmax/0.04)))
    X, K, lmu, lw, ly, lT, lY11, lF11 = DN.pieces(C, ts, yH, eps, 2.0, om, 1.0, kmax, N)
    lrho = np.log(dcode) + S0 + lmu + lw
    lZ0, lZ1 = lse(ly[0]), lse(ly[1]); lP0, lP1 = ly[0] - lZ0, ly[1] - lZ1; lw0, lw1 = lP0 - lrho, lP1 - lrho
    Sbh = -(np.exp(lP0)*(lP0 - lrho)).sum(); rows = []
    for x in XS:
        if np.isfinite(x):
            R = ExactResolvent(lrho, [lP0, lP1], Sbh + x); lJ0, lJ1 = R.logJ()
        else:
            lJ0, lJ1 = -0.5*lw0, -0.5*lw1
        r00 = np.exp(lse(lP0 + lw0 + 2*lJ0))
        r10 = np.exp(lse(lT - lZ0 + lP0 + 0.5*(lZ0 - lZ1) - lrho + lJ0 + lJ1))
        r11 = np.exp(lse(lY11 - lZ0 + lP1 - lrho + 2*lJ1))
        rc = DN.decoded(lrho, ly, lT, lY11, Sbh + x)
        m_e = float(np.real(1 - abs(r10)**2/(r00*r11))); m_c = float(np.real(1 - abs(rc[1])**2/(rc[0]*rc[2])))
        rows.append(dict(x=x if np.isfinite(x) else 'inf', margin_exact=m_e, margin_canonical=m_c, c1_exact=float(np.real(r00))))
        print(f"density C={C} omega={om} x={x:+6.1f}: 1-|rho10|^2/(rho00 rho11) exact {m_e:+.5f} canonical {m_c:+.5f}   c1(level 0) exact {np.real(r00):.4f}", flush=True)
    return dict(C=C, omega=om, rows=rows)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = {}
    if which in ('all', 'stopwatch'):
        res['stopwatch'] = [stopwatch(C) for C in (10.0, 2.0)]
        json.dump(res['stopwatch'], open(OUT + 'step3i_stopwatch_exact.json', 'w'), indent=1, default=float)
    if which in ('all', 'density'):
        res['density'] = [density_exact(C, om) for C in (10.0, 2.0) for om in (0.0, 0.4)]
        json.dump(res['density'], open(OUT + 'step3i_density_exact.json', 'w'), indent=1, default=float)
