"""Step 3d runs: (A) fixed mu = m/C, C -> infinity: exact -> R_0(mu);  (B) the separation at the couplings of Step 3b."""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3d_backreaction as B
ts = np.pi/4; LAB = [-2.347, -1.347, -0.347, 0.0, 0.312]; names = ['y_H-2', 'y_H-1', 'horizon', 'phi=0', '0.9 inner']
def ex(C, m, dys, ps=(1, 2)):
    X = B.Exact(C); k, _ = X.grid(m, ts); X.N = int(max(2400, min(7000, k[-1]/0.04))); return X.moments(ts, dys, m, ps)
out = {'A': [], 'B': []}; which = sys.argv[1] if len(sys.argv) > 1 else 'AB'
if 'A' in which:
    for mu in (0.25, 0.5, 1.0):
        for C in (10.0, 20.0, 40.0, 80.0):
            t = time.time(); M = ex(C, mu*C, LAB); r0 = np.array([B.R0(mu, ts, d) for d in LAB]); rc = B.R_cl(ts, np.array(LAB))
            row = dict(mu=mu, C=C, exact_over_Rcl=(M[1].real/rc).tolist(), R0_over_Rcl=(r0/rc).tolist(),
                       C_times_exact_over_R0_minus1=(C*(M[1].real/r0 - 1)).tolist(), C_relvar=(C*(M[2]/M[1]**2 - 1).real).tolist())
            out['A'].append(row)
            print(f"mu={mu} C={C:4.0f} ({time.time()-t:.0f}s): exact/Rcl " + " ".join(f"{v:.4f}" for v in row['exact_over_Rcl'])
                  + " | R0/Rcl " + " ".join(f"{v:.4f}" for v in row['R0_over_Rcl'])
                  + " | C(exact/R0-1) " + " ".join(f"{v:+.4f}" for v in row['C_times_exact_over_R0_minus1'])
                  + " | C relvar " + " ".join(f"{v:.4f}" for v in row['C_relvar']), flush=True)
if 'B' in which:
    for m in (1.0, 2.0, 4.0):
        for C in (160.0, 80.0, 40.0, 10.0, 4.0, 2.0, 1.0, 0.5):
            mu = m/C; M = ex(C, m, LAB); r0 = np.array([B.R0(mu, ts, d) for d in LAB]); rc = B.R_cl(ts, np.array(LAB))
            row = dict(m=m, C=C, mu=mu, kappa1=B.saddle(mu, ts)[1], total=(M[1].real/rc - 1).tolist(), classical=(r0/rc - 1).tolist(),
                       quantum=(M[1].real/r0 - 1).tolist(), relvar=((M[2]/M[1]**2 - 1).real).tolist())
            out['B'].append(row)
            print(f"m={m} C={C:5.1f} k1={row['kappa1']:.3f}: total " + " ".join(f"{v:+.4f}" for v in row['total']) + " | classical "
                  + " ".join(f"{v:+.4f}" for v in row['classical']) + " | quantum " + " ".join(f"{v:+.4f}" for v in row['quantum'])
                  + " | C*quantum " + " ".join(f"{C*v:+.4f}" for v in row['quantum']), flush=True)
json.dump(dict(ts=ts, labels=LAB, names=names, **out), open(f'03_phase3_quantum/results/step3d_fixed_labels_{which}.json', 'w'), indent=1)
