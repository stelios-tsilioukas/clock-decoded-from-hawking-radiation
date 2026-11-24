"""Step 2D analysis: collapse of deviations onto k/kappa = c/(6(S-S0)); test of the endurance formula."""
import json, numpy as np, sys
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
R = json.load(open('02_phase2_jt/results/step2d_runs.json'))
out = {'coefficients': {}, 'endurance': []}
for ts in sorted(set(round(r['ts_inst'], 2) for r in R)):
    pass
groups = {}
for r in R:
    key = round(np.arcsin(2*r['kappa']/r['phi_s']), 3); groups.setdefault(key, []).append(r)
print("coefficient = deviation/(k/kappa)  [rows: k, kappa]")
for key, rows in sorted(groups.items()):
    tab = []
    for r in rows:
        e = r['k']/r['kappa']
        tab.append([r['k'], r['kappa'], r['d1']/e, r['d2']/e, r['d3_rindler']/e, r['loop_dev_max']/e,
                    (r['tau_inner'] - r['tau_inner_static'])/e])
    tab = np.array(tab); out['coefficients'][str(key)] = tab.tolist()
    print(f"tau_s = {key}:  cols d1, d2, 2kap*dY_end, loop_dev, d_inner")
    for t in tab: print("   k=%-7g kap=%.3f  %+8.4f %+8.4f %+8.4f %8.4f %+8.4f" % tuple(t))
# endurance formula  tau_x = eta/(1+eta) * (k/kappa) * tan(ts)
Ms = {}
for r in R:
    if 'tau_cross_QES' not in r: continue
    k = r['k']; M = Ms.setdefault(k, LogJT(k)); Y, eta = M.qes(r['y_cross_QES'])
    ts = r['ts_inst']; kap = M.kappa(r['y_cross_QES'])
    pred = eta/(1 + eta)*(k/kap)*np.tan(ts)
    out['endurance'].append(dict(k=k, kappa=r['kappa'], eta=eta, tau_x=r['tau_cross_QES'], pred=pred))
    print(f"k={k:<7g} kap={r['kappa']:.3f} eta={eta:+.4f} tau_x={r['tau_cross_QES']:+.4e} formula={pred:+.4e} ratio={r['tau_cross_QES']/pred:.4f}")
json.dump(out, open('02_phase2_jt/results/step2d_analysis.json', 'w'), indent=1)
