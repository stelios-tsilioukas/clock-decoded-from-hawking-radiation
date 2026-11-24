"""Item 1 (all e^{-2 S0} terms) re-run at fixed physical parameters (Step 3d): black hole with the clock in it has
S - S0 = 2 pi k1, physical release depth tsp; labels at phi = 0, the true horizon and the release point."""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3d_backreaction as B
from step3c_all_terms import AllTerms
k1, m = float(sys.argv[1]), float(sys.argv[2]); tsp = np.pi/4
C, ts, x1 = B.phys_to_bare(k1, tsp, m); dH = -B.dy_horizon(tsp); dS = 2*np.arctanh(np.tan(tsp/2))
xg, wg = np.polynomial.legendre.leggauss(5); YQ = dH*(1 + xg)/2; WQ = dH*wg/2
X = AllTerms(C, ts, m, p=1.0); Nn = X.norm(); dn = (Nn['ab'] + Nn['Ga'] + Nn['Gb'])/Nn['disk']; c2 = (4*np.pi**2)**2
def c_at(d):
    T = X.run(d/x1); M = T['disk']; return c2*(sum(v for k, v in T.items() if k != 'disk')/M - dn), c2*T['bc']/M, M
t = time.time(); res = {nm: c_at(d)[:2] for nm, d in (('phi0', 0.0), ('horizon', dH), ('release', dS))}
q = [c_at(d) for d in YQ]; Md = np.array([v[2] for v in q]); cq = np.array([v[0] for v in q])
dT = float((cq*Md*WQ).sum()/(Md*WQ).sum())
out = dict(k1=k1, S_minus_S0=2*np.pi*k1, m=m, C_bare=C, ts_bare=ts, kappa1=x1, c=res, dT_rel_interior=dT)
print(json.dumps(out), f"{time.time()-t:.0f}s", flush=True)
json.dump(out, open(f'03_phase3_quantum/results/step3d_all_terms_phys_k{k1:g}_m{m:g}.json', 'w'), indent=1, default=float)
