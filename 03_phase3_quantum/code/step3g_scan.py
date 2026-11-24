"""Step 3g: decoded tick-rate mean and spread across the Page transition (x = log k - S_BH)."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3g_decoding as D, step3d_backreaction as B
ts = np.pi/4; m = 2.0; S0 = 8.0
out = {}
for C in (10.0, 2.0):
    X = D.Decoder(C, ts, m, S0); _, _, Sbh, _ = X.filter(0.0)
    # labels: true horizon and phi = 0 in bare variables (from the saddle temperature, Step 3d)
    x1 = B.saddle(m/C, ts)[1]; dH = B.dy_horizon(x1*ts)/x1; labs = {'horizon': dH, 'phi=0': 0.0}
    rows = []
    print(f"C={C}: S_BH - S0 = {Sbh - S0:.2f}  (labels: horizon dy={dH:.3f})")
    for x in (-6, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 6):
        mo, c1, _, sk = X.decoded_moments(list(labs.values()), [1, 2], Sbh + x)
        r = dict(x=x, c1=c1, s_k=sk)
        for j, nm in enumerate(labs):
            (p1, d1), (p2, d2) = mo[1][j], mo[2][j]
            r[nm] = dict(mean_phys=p1.real, mean_dec=d1.real, relvar_phys=(p2/p1**2 - 1).real, relvar_dec=(d2/d1**2 - 1).real)
        rows.append(r)
        h, f = r['horizon'], r['phi=0']
        print(f"  x={x:+5.1f}: c1={c1:.4f} s_k={sk:6.2f} | horizon: <R>_dec/<R>_phys-1={h['mean_dec']/h['mean_phys']-1:+.2e}  relvar phys {h['relvar_phys']:.4f} dec {h['relvar_dec']:.4f}"
              f" | phi=0: {f['mean_dec']/f['mean_phys']-1:+.2e}  relvar {f['relvar_phys']:.4f} -> {f['relvar_dec']:.4f}", flush=True)
    out[str(C)] = dict(S_BH_minus_S0=Sbh - S0, labels=labs, rows=rows)
json.dump(out, open('03_phase3_quantum/results/step3g_scan.json', 'w'), indent=1)
