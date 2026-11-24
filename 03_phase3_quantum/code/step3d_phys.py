"""Step 3d: the exact tick rate at fixed *physical* parameters -- entropy S - S0 = 2 pi k1 of the black hole with the clock
in it, physical release depth tsp, clock mass m -- in units of the physical temperature (kappa_phys = 1)."""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3d_backreaction as B
tsp = float(sys.argv[1]) if len(sys.argv) > 1 else np.pi/4; tag = sys.argv[2] if len(sys.argv) > 2 else 'pi4'
cases = eval(sys.argv[3]) if len(sys.argv) > 3 else [(40, 1.5), (40, 2), (40, 4), (40, 8), (10, 1.5), (10, 2), (10, 3), (10, 4), (6, 1.5), (6, 2), (5, 1.5), (5, 2), (4, 1.5)]
dH = B.dy_horizon(tsp); LAB = np.array([dH - 2, dH - 1, dH, 0.0, -0.9*dH]); names = ['y_H-2', 'y_H-1', 'horizon', 'phi=0', '0.9 inner']
xg, wg = np.polynomial.legendre.leggauss(8); YQ = dH*(1 - xg)/2; WQ = -dH*wg/2          # horizon -> phi = 0
rows = []
for k1, m in cases:
    t = time.time(); C, ts, x1 = B.phys_to_bare(k1, tsp, m)
    X = B.Exact(C); k, _ = X.grid(m, ts); X.N = int(max(2400, min(7000, k[-1]/0.04)))
    M = X.moments(ts, LAB/x1, m, (1, 2)); Mq = X.moments(ts, YQ/x1, m, (1,))
    Rp = M[1].real/x1; rc = B.R_cl(tsp, LAB); T = float((Mq[1].real/x1*WQ).sum())
    row = dict(k1=k1, S_minus_S0=2*np.pi*k1, m=m, m_over_mmax=m/B.m_max(k1, tsp), C_bare=C, ts_bare=ts, kappa1=x1,
               quantum=(Rp/rc - 1).tolist(), k1_times_quantum=(k1*(Rp/rc - 1)).tolist(),
               sd_over_mean=np.sqrt((M[2]/M[1]**2 - 1).real).tolist(), k1_relvar=(k1*(M[2]/M[1]**2 - 1).real).tolist(),
               T_H_to_phi0=T, T_classical=tsp, k1_times_dT_over_T=k1*(T/tsp - 1))
    rows.append(row)
    print(f"S-S0={2*np.pi*k1:6.1f} m={m} (m/mmax={row['m_over_mmax']:.2f}; bare C={C:.3f} ts={ts:.3f} k1/C={x1:.3f}; {time.time()-t:.0f}s): k1*quantum "
          + " ".join(f"{v:+.4f}" for v in row['k1_times_quantum']) + " | k1*relvar " + " ".join(f"{v:.4f}" for v in row['k1_relvar'])
          + f" | T(H->phi0)/tsp-1 = {T/tsp-1:+.5f}  (x k1: {row['k1_times_dT_over_T']:+.4f})", flush=True)
json.dump(dict(tsp=tsp, labels=LAB.tolist(), names=names, rows=rows),
          open(f'03_phase3_quantum/results/step3d_physical_{tag}.json', 'w'), indent=1)
