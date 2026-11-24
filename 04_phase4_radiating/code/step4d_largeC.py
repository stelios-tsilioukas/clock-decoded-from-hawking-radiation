"""
Step 4d (audit item M3): large-C runs of the decoded infalling stopwatch after the lse fix (an all -inf slice of the
log-sum-exp returned nan; 21 Nov 2026).  (1) bare protocol (tau_s = pi/4, Delta = 2) at C = 10, 20, 40: large-C limit of
C x (decoded - one-sided); (2) physical variables k1 = 10, 20, 40 (ts' = pi/4, m = 2): k1 x decoder / k1 x quantum;
(3) grid check at C = 40 (N = 2800 vs 4000).
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step4b_decorated import Decorated, vis
from step3h_replica import lse
from step3k_split import reading_cl
OUT = '04_phase4_radiating/results/'
om, h, eps, De, ee = 0.2, 1.0, 0.05, 1.0, 0.0125

def readings(C, ts, x1, labs, N=None, xs=(np.inf,), kfac=1.0):
    kmax = kfac*(C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/ee) + 10); N = N or int(min(2800, max(1400, kmax/0.05)))
    S = Decorated(C, ts, 2.0, om, h, eps, kmax, N); S.setup_emission(De, ee)
    filt = {x: S.filter_e(x) for x in xs}; out = {}
    for nm, y in labs.items():
        phys, Y, _ = S.elements_e(y)
        out[nm] = dict(phys=vis(phys, om), dec={str(x): vis({k: lse(v + filt[x]) for k, v in Y.items()}, om) for x in xs})
    return out, N, kmax

if __name__ == "__main__":
    mode = sys.argv[1]; res = []
    if mode == 'bare':
        for C in (10., 20., 40.):
            t0 = time.time(); x1 = B.saddle(2.0/C, np.pi/4)[1]; yH = B.dy_horizon(x1*np.pi/4)/x1
            r, N, km = readings(C, np.pi/4, x1, {'horizon': yH, '0.9 inner': -0.9*yH}, xs=(0, 12, 20, np.inf))
            for nm, v in r.items():
                print(f"bare C={C:4.0f} {nm:9s}: phys {v['phys'][1]:+.6f} dec deep {v['dec']['inf'][1]:+.6f} C*(dec-phys) {C*(v['dec']['inf'][1]-v['phys'][1]):+.5f}"
                      f" | 1-|V| dec {1-v['dec']['inf'][0]:.1e} | x=0,12,20: " + " ".join(f"{v['dec'][x][1]:+.5f}" for x in ('0','12','20')) + f"  N={N} [{time.time()-t0:.0f}s]", flush=True)
            res.append(dict(C=C, x1=x1, N=N, kmax=km, readings=r))
    elif mode == 'phys':
        for k1 in (40.,):
            t0 = time.time(); C, ts, x1 = B.phys_to_bare(k1, np.pi/4, 2.0); yHp = B.dy_horizon(np.pi/4)
            labs = {'y_H-1': yHp - 1.0, 'horizon': yHp, '0.9 inner': -0.9*yHp}
            r, N, km = readings(C, ts, x1, {k: v/x1 for k, v in labs.items()})
            cl = {k: reading_cl(np.pi/4, v, 2.0, om, h, x1*eps)[1] for k, v in labs.items()}
            for a, b in (('y_H-1', 'horizon'), ('horizon', '0.9 inner')):
                c = cl[b] - cl[a]; p = r[b]['phys'][1] - r[a]['phys'][1]; d = r[b]['dec']['inf'][1] - r[a]['dec']['inf'][1]
                print(f"phys k1={k1:.0f} (C={C:.3f}, m/m_max={2/B.m_max(k1, np.pi/4):.2f}) {a}->{b}: classical {c:+.6f} one-sided {p:+.6f} decoded {d:+.6f}"
                      f" | k1*quantum {k1*(p-c)/abs(c):+.4f} k1*decoder {k1*(d-p)/abs(c):+.4f} ratio {(d-p)/(p-c):+.3f}  N={N} [{time.time()-t0:.0f}s]", flush=True)
            res.append(dict(k1=k1, C=C, x1=x1, N=N, kmax=km, readings=r, classical=cl))
    elif mode == 'grid':
        C = 40.; x1 = B.saddle(2.0/C, np.pi/4)[1]; yH = B.dy_horizon(x1*np.pi/4)/x1
        for N in (2800, 4000):
            t0 = time.time(); r, _, km = readings(C, np.pi/4, x1, {'horizon': yH}, N=N)
            v = r['horizon']; print(f"grid C=40 N={N}: phys {v['phys'][1]:+.7f} dec {v['dec']['inf'][1]:+.7f} C*(dec-phys) {C*(v['dec']['inf'][1]-v['phys'][1]):+.5f} [{time.time()-t0:.0f}s]", flush=True)
            res.append(dict(N=N, readings=r))
    json.dump(res, open(OUT + f'step4d_largeC_{mode}.json', 'w'), indent=1, default=float)
