"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.
Step 3h: frozen stopwatch decoded across the Page transition (C = 10 and 2), horizon and 0.9-inner labels."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_stopwatch as SW, step3d_backreaction as B
ts = np.pi/4; D = 2.0; h = 1.0; eps = 0.05; om = 0.2
xs = [-12, -8, -4, -2, 0, 2, 4, 8, 12, 20, np.inf]; res = {}
for C in (10.0, 2.0):
    x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2600, max(1400, kmax/0.04)))
    S = SW.Stopwatch(C, ts, D, om, h, eps, kmax, N); rows = {}
    for nm, y in (('horizon', yH), ('0.9 inner', -0.9*yH)):
        Vp, Vds = S.visibility(y, xs)
        rows[nm] = dict(phys=[abs(Vp), np.angle(Vp)/om], dec=[[abs(v), np.angle(v)/om] for v in Vds])
        for x, v in zip(xs, Vds):
            print(f"C={C} {nm:9s} x={x:+6.1f}: |V| dec {abs(v):.5f} (phys {abs(Vp):.5f})  reading dec {np.angle(v)/om:+.5f} (phys {np.angle(Vp)/om:+.5f}, dec/phys-1 = {np.angle(v)/np.angle(Vp)-1:+.5f})", flush=True)
    res[str(C)] = rows
json.dump(res, open('03_phase3_quantum/results/step3h_stopwatch_scan.json', 'w'), indent=1, default=float)
