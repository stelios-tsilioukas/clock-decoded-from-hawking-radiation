"""Step 3e: exact matter-clock reading in physical variables (black hole with the clock in it: S - S0 = 2 pi k1,
physical release depth tsp, mean clock mass Dbar).  Reading shift  dtau = [arg A_omega - omega tau_phys]/omega ;
log 'visibility'  ln V = ln|A_omega/A_0| - (1/2) ln[N(Dbar - w/2) N(Dbar + w/2)/N(Dbar)^2]."""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3e_matter_clock as M, step3d_backreaction as B
tsp = float(sys.argv[1]); cases = eval(sys.argv[2]); tag = sys.argv[3]
dH = B.dy_horizon(tsp); LAB = np.array([dH - 2, dH - 1, dH, 0.0, -0.9*dH]); names = ['y_H-2', 'y_H-1', 'horizon', 'phi=0', '0.9 inner']
ol = [M.lnZhat_coeffs_bar(tsp, d) for d in LAB]
rows = []
for k1, Db, h, om in cases:
    t = time.time(); C, ts, x1 = B.phys_to_bare(k1, tsp, Db); X = M.Exact(C, N=4000)
    wa, wb, wc = (h - om)/2, (h + om)/2, Db - h/2
    L1 = X.log_triangle(ts, LAB/x1, wa, wb, wc, m_grid=Db); L0 = X.log_triangle(ts, LAB/x1, h/2, h/2, wc, m_grid=Db)
    lN = lambda l: X.log_two_point(l, 2*ts, m_grid=Db, ts=ts).real
    lnV = (L1.real - L0.real) - 0.5*(lN(Db - om/2) + lN(Db + om/2) - 2*lN(Db))
    dtau = (L1.imag - om*M.tau_phys(tsp, LAB))/om
    pred_dtau = np.array([(c['w'] + h*c['wh']).imag for c in ol]); pred_V = np.array([c['ww'].real for c in ol])
    row = dict(k1=k1, S_minus_S0=2*np.pi*k1, Dbar=Db, h=h, omega=om, m_over_mmax=Db/B.m_max(k1, tsp), C_bare=C, ts_bare=ts, kappa1=x1,
               k1_dtau=(k1*dtau).tolist(), oneloop_dtau=pred_dtau.tolist(), k1_lnV_over_w2=(k1*lnV/om**2).tolist(), oneloop_lnV=pred_V.tolist())
    rows.append(row)
    print(f"S-S0={2*np.pi*k1:6.1f} Dbar={Db} h={h} w={om} (m/mmax={row['m_over_mmax']:.2f}, {time.time()-t:.0f}s) | k1*dtau "
          + " ".join(f"{v:+.4f}" for v in row['k1_dtau']) + " [1-loop " + " ".join(f"{v:+.4f}" for v in pred_dtau)
          + "] | k1*lnV/w^2 " + " ".join(f"{v:+.4f}" for v in row['k1_lnV_over_w2']) + " [1-loop " + " ".join(f"{v:+.4f}" for v in pred_V) + "]", flush=True)
json.dump(dict(tsp=tsp, labels=LAB.tolist(), names=names, rows=rows),
          open(f'03_phase3_quantum/results/step3e_physical_{tag}.json', 'w'), indent=1)
