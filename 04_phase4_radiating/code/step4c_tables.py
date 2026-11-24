"""
Step 4c (audit items M7 + M8): convention-free decoded intervals and the twin-loop phase of the INFALLING clock,
in physical variables.

Physical variables (Step 3d): S - S0 = 2 pi k1 (black hole with the clock in it), release depth ts' (sin ts' = phi_h/phi_s),
clock mass m = Delta; bare (C, ts) from B.phys_to_bare.  Labels are physical: boundary stop times in units of the actual
beta/2pi, measured from the phi = 0 label: y_H - 1, horizon, phi = 0, 0.9 inner (= -0.9 x horizon label).
Classical reference: fixed-geometry stopwatch reading (Step 3k block formula) at ts', physical labels, pulse width kappa_1 eps.
Decoded: radiating model of Step 4b (hub/gap, Delta_e, eps_e), exact-resolvent filter at x, x = inf deep post-Page.
Interval(a -> b) = reading(b) - reading(a), same decoder for both stop events: free of the logical-phase convention.
Twin loop (boundary station): Phi_loop/omega = Delta tau_clock(a -> b) - Delta t_station(a -> b), Delta t_station = label
difference (station clock normalised to boundary time in units of beta/2pi).  The station term is classical and exact.
"""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
sys.path.insert(0, '04_phase4_radiating/code')
import step3d_backreaction as B
from step3k_split import reading_cl
from step4b_decorated import Decorated, vis
from step3h_replica import lse
OUT = '04_phase4_radiating/results/'

def run(k1, tsp=np.pi/4, m=2.0, om=0.2, h=1.0, eps=0.05, De=1.0, eps_e=0.0125, xs=(0, 12, 20, np.inf)):
    C, ts, x1 = B.phys_to_bare(k1, tsp, m)
    yHp = B.dy_horizon(tsp)
    labs = {'y_H-1': yHp - 1.0, 'horizon': yHp, 'phi=0': 0.0, '0.9 inner': -0.9*yHp}
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps_e) + 10; N = int(min(2800, max(1400, kmax/0.05)))
    t0 = time.time(); S = Decorated(C, ts, m, om, h, eps, kmax, N); S.setup_emission(De, eps_e)
    filt = {x: S.filter_e(x) for x in xs}
    read = {}
    for nm, lab in labs.items():
        phys, Y, _ = S.elements_e(lab/x1)
        read[nm] = dict(label=lab, classical=reading_cl(tsp, lab, m, om, h, x1*eps)[1], phys=vis(phys, om),
                        dec={str(x): vis({k: lse(v + filt[x]) for k, v in Y.items()}, om) for x in xs})
        print(f"  k1={k1} {nm:9s} label {lab:+.4f}: classical {read[nm]['classical']:+.5f} one-sided {read[nm]['phys'][1]:+.5f} "
              f"(|V| {read[nm]['phys'][0]:.5f}) decoded deep {read[nm]['dec']['inf'][1]:+.5f} (|V| {read[nm]['dec']['inf'][0]:.6f})"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    return dict(k1=k1, C=C, ts=ts, kappa1=x1, tsp=tsp, m=m, m_over_mmax=m/B.m_max(k1, tsp), om=om, h=h, eps=eps,
                De=De, eps_e=eps_e, N=N, readings=read)

def table(r, pairs):
    rd = r['readings']; k1 = r['k1']; rows = []
    print(f"\nS - S0 = 2 pi x {k1:g}, ts' = pi/4, m/m_max = {r['m_over_mmax']:.2f} (bare C = {r['C']:.3f}, kappa1 = {r['kappa1']:.4f}),"
          f" Delta_e = {r['De']}, eps_e = {r['eps_e']}")
    print(f"{'interval':>22} | {'classical':>9} {'one-sided':>9} {'decoded':>9} | {'k1*quant':>8} {'k1*dec':>8} {'ratio':>6} |"
          f" dec/one-sided - 1 at x = 0, 12, 20 | twin loop /omega: classical one-sided decoded")
    for a, b in pairs:
        cl = rd[b]['classical'] - rd[a]['classical']; ph = rd[b]['phys'][1] - rd[a]['phys'][1]
        de = {x: rd[b]['dec'][x][1] - rd[a]['dec'][x][1] for x in rd[a]['dec']}
        dt = rd[b]['label'] - rd[a]['label']
        q = k1*(ph - cl)/abs(cl); d = k1*(de['inf'] - ph)/abs(cl)
        tr = [de[x]/ph - 1 for x in ('0', '12', '20')]
        loop = (cl - dt, ph - dt, de['inf'] - dt)
        print(f"{a+' -> '+b:>22} | {cl:+9.5f} {ph:+9.5f} {de['inf']:+9.5f} | {q:+8.4f} {d:+8.4f} {d/q if q else np.nan:+6.3f} |"
              f" {tr[0]:+.4f} {tr[1]:+.4f} {tr[2]:+.5f} | {loop[0]:+.5f} {loop[1]:+.5f} {loop[2]:+.5f}", flush=True)
        rows.append(dict(pair=[a, b], station=dt, classical=cl, one_sided=ph, decoded={x: v for x, v in de.items()},
                         k1_rel_quantum=q, k1_rel_decoder=d, decoder_over_quantum=d/q if q else None,
                         transition=dict(zip(('0', '12', '20'), tr)), twin_loop_over_omega=dict(classical=loop[0], one_sided=loop[1], decoded=loop[2])))
    return rows

if __name__ == "__main__":
    k1s = [float(v) for v in sys.argv[1].split(',')] if len(sys.argv) > 1 else [10., 20.]
    pairs = [('y_H-1', 'horizon'), ('horizon', 'phi=0'), ('phi=0', '0.9 inner'), ('horizon', '0.9 inner'), ('y_H-1', '0.9 inner')]
    out = []
    for k1 in k1s:
        r = run(k1); r['table'] = table(r, pairs); out.append(r)
        json.dump(out, open(OUT + 'step4c_tables.json', 'w'), indent=1, default=float)
