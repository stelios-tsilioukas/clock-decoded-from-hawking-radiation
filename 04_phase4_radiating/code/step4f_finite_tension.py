"""
Step 4f (follow-up (h), 21 Nov 2026): the decoder correction at finite brane tension.
PSSY brane factor: each brane segment on a region of momentum k carries y_mu(k) = |Gamma(mu - 1/2 + i k)|^2 / Gamma(mu - 1/2)^2.
In the one-sided state and in every threaded replica pair the brane sits in the brane-side region D of the clock chord,
so the only change is colD -> colD + log y_mu(D) (which also enters F_chi, hence the reference gaps Gam_ss).
Deep post-Page (filter G = 1). Labels are BARE boundary times; mu = inf reproduces Step 4b/4c exactly.
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code'); sys.path.insert(0, '04_phase4_radiating/code')
from scipy.special import loggamma, gammaln
import step3d_backreaction as B
from step3h_replica import lse
from step4b_decorated import Decorated, vis

def log_ymu(mu, K):
    if not np.isfinite(mu): return np.zeros_like(K)
    return 2*loggamma(mu - 0.5 + 1j*K).real - 2*gammaln(mu - 0.5)

class FiniteTension(Decorated):
    def set_tension(self, mu):
        K = self.X.k; self.mu_brane = mu
        if not hasattr(self, 'colD0'): self.colD0 = self.colD.copy()
        self.colD = self.colD0 + log_ymu(mu, K)[:, None]
        self.lFx = lse(self.lgx + self.colD, axis=0)
        if hasattr(self, '_memo'): pass           # triangles logI do not depend on the brane (D is external)

def readings(S, y, om):
    phys, Y, Y0 = S.elements_e(y)
    d = vis({k: lse(v) for k, v in Y.items()}, om); p = vis(phys, om); y0 = vis({k: lse(v) for k, v in Y0.items()}, om)
    return p, d, y0

if __name__ == "__main__":
    C = float(sys.argv[1]); mus = [float(v) if v != 'inf' else np.inf for v in sys.argv[2].split(',')]
    ys = [float(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    ts, D, h, eps, om, De, ee = np.pi/4, 2.0, 1.0, 0.05, 0.2, 1.0, 0.0125
    x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    if ys is None: ys = [yH - 1/x1, yH, -0.9*yH]
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/ee) + 10; N = int(min(2800, max(1400, kmax/0.05)))
    S = FiniteTension(C, ts, D, om, h, eps, kmax, N); out = []
    for mu in mus:
        t0 = time.time(); S.set_tension(mu); S.setup_emission(De, ee)
        for y in ys:
            p, d, y0 = readings(S, y, om)
            row = dict(C=C, mu=mu, y=y, yH=yH, phys=p, dec=d, dec_noemission=y0)
            out.append(row)
            print(f"C={C:g} mu={mu:g} y={y:+.4f}: one-sided {p[1]:+.6f} (|V| {p[0]:.5f}) dec {d[1]:+.6f} (|V| {d[0]:.6f}) "
                  f"C*(dec-1s) {C*(d[1]-p[1]):+.5f} | no-emission dec C*diff {C*(y0[1]-p[1]):+.5f}  [{time.time()-t0:.0f}s]", flush=True)
    tag = sys.argv[4] if len(sys.argv) > 4 else ''
    json.dump(out, open(f'04_phase4_radiating/results/step4f_finite_tension_C{int(C)}{tag}.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- physical variables at finite brane tension
from scipy.optimize import fsolve, brentq
def saddle_mu(m_over_C, ts, mub_over_C):
    """two-region saddle (x0 = brane side D, x1 = clock side) with brane factor y_mu on D (large-argument digamma)."""
    mu = m_over_C
    def eqs(v):
        x0, x1 = v
        A = np.arctan2(x0 + x1, mu); Bq = np.arctan2(x0 - x1, mu)
        br = 0.0 if not np.isfinite(mub_over_C) else np.arctan2(x0, mub_over_C)
        return [x0*(np.pi - ts) - (np.pi - A - Bq - br), x1*ts - (np.pi - A + Bq)]
    x0i, x1i = B.saddle(mu, ts)
    x = fsolve(eqs, [x0i, x1i], xtol=1e-13); assert max(abs(np.array(eqs(x)))) < 1e-10
    return float(x[0]), float(x[1])

def run_phys(C, mus, ts_phys=None, D=2.0, h=1.0, eps=0.05, om=0.2, De=1.0, ee=0.0125, tag=''):
    if ts_phys is None:
        x1i = B.saddle(D/C, np.pi/4)[1]; ts_phys = x1i*np.pi/4          # physical depth of the mu = inf protocol point
    out = []
    for mu in mus:
        mubC = mu/C
        f = lambda t: saddle_mu(D/C, t, mubC)[1]*t - ts_phys
        tsb = brentq(f, 0.3, 1.2, xtol=1e-12); x0, x1 = saddle_mu(D/C, tsb, mubC)
        yH = B.dy_horizon(ts_phys)/x1
        labs = {'H-1': yH - 1/x1, 'H': yH, '0.9in': -0.9*yH}
        kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/ee) + 10; N = int(min(2800, max(1400, kmax/0.05)))
        t0 = time.time()
        S = FiniteTension(C, tsb, D, om, h, eps, kmax, N); S.set_tension(mu); S.setup_emission(De, ee)
        row = dict(C=C, mu=mu, ts_bare=tsb, x0=x0, x1=x1, k1=x1*C, ts_phys=ts_phys, labels=labs, r={})
        for nm, y in labs.items():
            p, d, y0 = readings(S, y, om); row['r'][nm] = dict(phys=p, dec=d, dec_noem=y0)
        k1 = x1*C; r = row['r']
        dd = {nm: k1*(r[nm]['dec'][1] - r[nm]['phys'][1]) for nm in labs}
        Dp = {nm: k1*(r[nm]['dec_noem'][1] - r[nm]['phys'][1]) for nm in labs}
        row['k1_dec_minus_1s'] = dd; row['k1_Dpart'] = Dp; row['k1_gappart'] = {nm: dd[nm] - Dp[nm] for nm in labs}
        print(f"C={C:g} mu={mu:g} (mu/C={mubC:g}): bare ts={tsb:.5f} x0={x0:.4f} x1={x1:.4f} k1={k1:.3f} | "
              + " | ".join(f"{nm}: 1s {r[nm]['phys'][1]:+.6f} dec {r[nm]['dec'][1]:+.6f} k1*d {dd[nm]:+.5f} (D {Dp[nm]:+.4f}, gap {dd[nm]-Dp[nm]:+.4f}) 1-|V| 1s {1-r[nm]['phys'][0]:.2e} dec {1-r[nm]['dec'][0]:.1e}" for nm in labs)
              + f"  [{time.time()-t0:.0f}s]", flush=True)
        out.append(row)
        json.dump(out, open(f'04_phase4_radiating/results/step4f_phys_C{int(C)}{tag}.json', 'w'), indent=1, default=float)
    return out

# ---------------------------------------------------------------- Step 4c protocol (physical k1, tau_s', m) at finite tension
from step3k_split import reading_cl
from scipy.special import digamma
def phys_to_bare_mu(k1, tsp, m, mub, exact_brane=False):
    """as B.phys_to_bare, with the brane factor in the brane-side saddle: the junction fixes k0, A, B (clock side unchanged);
    the brane-side saddle gives t0 = (pi - A - B - atan(k0/mub))/k0 (large-argument digamma, consistent with saddle_mu)."""
    E0 = k1**2 - 2*m*k1/np.tan(tsp) - m**2; k0 = np.sqrt(E0)
    A = np.arctan2(k0 + k1, m); Bq = np.arctan2(k0 - k1, m)
    if not np.isfinite(mub): br = 0.0
    elif exact_brane: br = float(np.imag(digamma(mub - 0.5 + 1j*k0)))      # exact d/dk0 of log y_mu / 2
    else: br = np.arctan2(k0, mub)
    t0 = (np.pi - A - Bq - br)/k0; bM = t0 + tsp/k1; C = np.pi/bM; x1 = k1/C
    return C, tsp/x1, x1, k0

def run4c(k1, r_mu, tsp=np.pi/4, m=2.0, om=0.2, h=1.0, eps=0.05, De=1.0, eps_e=0.0125, exact_brane=False):
    mub = r_mu*k1
    C, ts, x1, k0 = phys_to_bare_mu(k1, tsp, m, mub, exact_brane)
    if not exact_brane:
        x0c, x1c = saddle_mu(m/C, ts, mub/C); assert abs(x1c - x1) < 1e-8, (x1c, x1)
    yHp = B.dy_horizon(tsp)
    labs = {'y_H-1': yHp - 1.0, 'horizon': yHp, 'phi=0': 0.0, '0.9 inner': -0.9*yHp}
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps_e) + 10; N = int(min(2800, max(1400, kmax/0.05)))
    t0 = time.time(); S = FiniteTension(C, ts, m, om, h, eps, kmax, N); S.set_tension(mub); S.setup_emission(De, eps_e)
    read = {}
    for nm, lab in labs.items():
        phys, Y, Y0 = S.elements_e(lab/x1)
        read[nm] = dict(label=lab, classical=reading_cl(tsp, lab, m, om, h, x1*eps)[1], phys=vis(phys, om),
                        dec=vis({k: lse(v) for k, v in Y.items()}, om), dec_noem=vis({k: lse(v) for k, v in Y0.items()}, om))
    pairs = [('y_H-1', 'horizon'), ('horizon', 'phi=0'), ('phi=0', '0.9 inner'), ('horizon', '0.9 inner'), ('y_H-1', '0.9 inner')]
    rows = []
    for a, b in pairs:
        cl = read[b]['classical'] - read[a]['classical']; ph = read[b]['phys'][1] - read[a]['phys'][1]
        de = read[b]['dec'][1] - read[a]['dec'][1]; d0 = read[b]['dec_noem'][1] - read[a]['dec_noem'][1]
        q = k1*(ph - cl)/abs(cl); d = k1*(de - ph)/abs(cl); dD = k1*(d0 - ph)/abs(cl)
        rows.append(dict(pair=[a, b], classical=cl, one_sided=ph, decoded=de, k1_quantum=q, k1_decoder=d, k1_Dpart=dD, k1_gappart=d - dD,
                         ratio=d/q if q else None))
    res = dict(k1=k1, r_mu=r_mu, mu=mub, exact_brane=exact_brane, C=C, ts=ts, kappa1=x1, k0=k0, N=N, readings=read, intervals=rows,
               purity={nm: dict(one_sided=1 - read[nm]['phys'][0], decoded=1 - read[nm]['dec'][0]) for nm in labs})
    print(f"k1={k1:g} mu={'inf' if not np.isfinite(mub) else f'{mub:g}'} (mu/k1={r_mu:g}): bare C={C:.4f} ts={ts:.5f} kappa1={x1:.5f}  [{time.time()-t0:.0f}s]", flush=True)
    for rw in rows:
        print(f"   {rw['pair'][0]+' -> '+rw['pair'][1]:>22}: classical {rw['classical']:+.5f} one-sided {rw['one_sided']:+.5f} decoded {rw['decoded']:+.5f} | "
              f"k1*quantum {rw['k1_quantum']:+.4f} k1*decoder {rw['k1_decoder']:+.5f} (D {rw['k1_Dpart']:+.4f}, gap {rw['k1_gappart']:+.4f}) ratio {rw['ratio']:+.3f}", flush=True)
    print("   purity 1-|V|: " + "  ".join(f"{nm}: {v['one_sided']:.2e}/{v['decoded']:.1e}" for nm, v in res['purity'].items()), flush=True)
    return res
