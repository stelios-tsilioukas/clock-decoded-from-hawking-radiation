"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.
Step 3h: is the decoder offset a convention?  Compare decoded and physical single-event readings at several labels;
the interval (difference of readings at two events) is the convention-free observable."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_replica as H, step3d_backreaction as B
ts = np.pi/4; D = 2.0; h = 1.0; out = []
for C in (10.0, 20.0, 40.0):
    X = H.Replica(C, ts, D); x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    labs = {'y_H-1': yH - 1/x1, 'horizon': yH, 'phi=0': 0.0, '0.9 inner': -0.9*yH}
    ly0 = X.log_y(D)
    for om in (0.1, 0.2):
        ly1 = X.log_y(D + om); res = {}
        for nm, dy in labs.items():
            lT0 = X.log_T(dy, h/2, h/2, D - h/2); lT1 = X.log_T(dy, (h + om)/2, (h - om)/2, D + (om - h)/2)
            _, Cp0, Cd0, _, _ = X.decode(lT0, ly0, ly0, np.inf); _, Cp1, Cd1, _, _ = X.decode(lT1, ly0, ly1, np.inf)
            res[nm] = dict(phys=(np.angle(Cp1) - np.angle(Cp0))/om, dec=(np.angle(Cd1) - np.angle(Cd0))/om)
        off = {nm: r['dec'] - r['phys'] for nm, r in res.items()}
        iv = {f"{a}->{b}": ((res[b]['dec'] - res[a]['dec']), (res[b]['phys'] - res[a]['phys']))
              for a, b in (('y_H-1', 'horizon'), ('horizon', 'phi=0'), ('horizon', '0.9 inner'))}
        out.append(dict(C=C, omega=om, readings=res, offsets=off, intervals=iv))
        print(f"C={C:4.0f} omega={om}: offsets dec-phys " + "  ".join(f"{nm} {v:+.5f}" for nm, v in off.items()))
        print("            intervals (decoded, physical, C x diff): " + "  ".join(f"{k}: {d:.5f} {p:.5f} {C*(d-p):+.4f}" for k, (d, p) in iv.items()), flush=True)
json.dump(out, open('03_phase3_quantum/results/step3h_interval.json', 'w'), indent=1, default=float)
