"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.
Step 3h: decoded tick-rate moments across the Page transition, omega = 0, filter on the shared region (cap [u_+, y]).
<R^p>_dec = s^p Re C_dec(p);  relvar_dec = <R^2>/<R>^2 - 1 must be >= 0 (decoded state is positive)."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_replica as H, step3d_backreaction as B
ts = np.pi/4; m = 2.0; out = {}
for C in (10.0, 2.0):
    X = H.Replica(C, ts, m); s = X.s; ly = X.log_y(m)
    x1 = B.saddle(m/C, ts)[1]; labs = {'horizon': B.dy_horizon(x1*ts)/x1, 'phi=0': 0.0, 'y_H-1': B.dy_horizon(x1*ts)/x1 - 1/x1}
    lT = {nm: {p: X.log_T(dy, p/2, p/2, m - p/2) for p in (1, 2)} for nm, dy in labs.items()}
    lZ = H.lse(ly); P = np.exp(ly - lZ); Sbh = -(P*(np.log(P) - X.lrho)).sum()
    rows = []; print(f"C={C}: S_BH - S0 = {Sbh - X.S0:.2f}")
    for x in (-12, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12, 20, np.inf):
        r = dict(x=x)
        for nm in labs:
            mo = {}
            for p in (1, 2):
                Cd, Cp, _, c1, sk = X.decode(lT[nm][p], ly, ly, Sbh + x)
                mo[p] = ((Cd*s**p).real, (Cp*s**p).real)
            r[nm] = dict(mean_dec=mo[1][0], mean_phys=mo[1][1], relvar_dec=mo[2][0]/mo[1][0]**2 - 1, relvar_phys=mo[2][1]/mo[1][1]**2 - 1)
        r['c1'] = c1; rows.append(r)
        h, f, e = r['horizon'], r['phi=0'], r['y_H-1']
        print(f"  x={x:+6.1f} c1={c1:.4f} | horizon: mean dec/phys-1 {h['mean_dec']/h['mean_phys']-1:+.4f}, relvar {h['relvar_dec']:.5f} (phys {h['relvar_phys']:.5f})"
              f" | phi=0: {f['mean_dec']/f['mean_phys']-1:+.4f}, {f['relvar_dec']:.5f} ({f['relvar_phys']:.5f})"
              f" | y_H-1: {e['mean_dec']/e['mean_phys']-1:+.4f}, {e['relvar_dec']:.5f} ({e['relvar_phys']:.5f})", flush=True)
    out[str(C)] = dict(labels=labs, S_BH_minus_S0=Sbh - X.S0, rows=[{k: (v if not isinstance(v, float) or np.isfinite(v) else 'inf') for k, v in rr.items()} for rr in rows])
json.dump(out, open('03_phase3_quantum/results/step3h_positivity.json', 'w'), indent=1, default=str)
