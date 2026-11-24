"""
Step 3k (audit item M5): split the one-sided stopwatch reading's O(1/C) shift into the clock's classical backreaction
and the genuine quantum part.

Classical stopwatch reading (fixed geometry, finite clock mass): 1d conformal block with externals psi_a (u_+),
V, V (y -+ i eps/2), psi_b (u_-) and chi exchange (Step 3f formula with unequal externals); reading = arg V/omega.
Step 3d: the clock's classical backreaction is a temperature shift kappa_1 (saddle of Step 3d) and nothing else, so the
backreacted classical reading is the fixed-geometry one at release depth ts' = kappa_1 ts, label kappa_1 y and pulse
width kappa_1 eps (all boundary times in units of the actual temperature).
Split:  exact - probe = [classical(ts', kappa_1 y) - classical(ts, y_probe)]  +  [exact - classical(ts', kappa_1 y)].
                         ^ clock backreaction (classical)                      ^ quantum (+ signal recoil, O(1/C))
Note: the reading at the phi = 0 label is 0 by time-reflection symmetry, so every reading is the horizon(or label) ->
phi = 0 interval in units where the classical heavy-clock reading is -Delta tau.
"""
import numpy as np, mpmath as mp, sys, time, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3d_backreaction as B, step3h_stopwatch as SW, step3e_matter_clock as M

def logG(ts, y, eps, Da, Db, Dx, h):
    u = [-ts + 0j, -eps/2 + 1j*y, eps/2 + 1j*y, ts + 0j]
    lx = lambda i, j: np.log(2*np.sin((u[j] - u[i])/2))
    D12 = Da - h; D34 = h - Db
    lz = lx(0, 1) + lx(2, 3) - lx(0, 2) - lx(1, 3)
    pre = -(Da + h)*lx(0, 1) - (h + Db)*lx(2, 3) + D12*(lx(1, 3) - lx(0, 3)) + D34*(lx(0, 3) - lx(0, 2))
    return pre + Dx*lz + np.log(complex(mp.hyp2f1(Dx - D12, Dx + D34, 2*Dx, mp.exp(lz))))

def reading_cl(ts, y, D, om, h, eps):
    Dx = D + om/2
    l01 = logG(ts, y, eps, D, D + om, Dx, h); l00 = logG(ts, y, eps, D, D, Dx, h); l11 = logG(ts, y, eps, D + om, D + om, Dx, h)
    V = np.exp(l01 - (l00 + l11)/2); return abs(V), np.angle(V)/om

def exact_sw(C, ts, D, om, h, eps):
    x1 = B.saddle(D/C, ts)[1]
    kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2800, max(1400, kmax/0.04)))
    return SW.Stopwatch(C, ts, D, om, h, eps, kmax, N), x1, N

def split(C, ts, D, om, h, eps, labels=('horizon', '0.9 inner')):
    S, x1, N = exact_sw(C, ts, D, om, h, eps); tsp = x1*ts
    yHp = B.dy_horizon(tsp)                        # physical horizon label (units of the actual temperature)
    yH0 = B.dy_horizon(ts)                         # probe horizon label
    phys = {'horizon': yHp, '0.9 inner': -0.9*yHp}; probe = {'horizon': yH0, '0.9 inner': -0.9*yH0}
    rows = {}
    for nm in labels:
        y = phys[nm]/x1
        Vp, _ = S.visibility(y, []); rex = np.angle(Vp)/om
        rpr = reading_cl(ts, probe[nm], D, om, h, eps)[1]
        rbr = reading_cl(tsp, phys[nm], D, om, h, x1*eps)[1]          # pulse width also in units of the actual temperature
        rows[nm] = dict(exact=rex, probe=rpr, backreacted=rbr, k1=C*x1,
                        C_total=C*(rex - rpr), C_classical=C*(rbr - rpr), C_quantum=C*(rex - rbr),
                        k1_rel_quantum=C*x1*(rex - rbr)/abs(rbr), absV_exact=abs(Vp))
    return dict(C=C, ts=ts, tsp=tsp, kappa1=x1, D=D, om=om, h=h, eps=eps, N=N, rows=rows)

def show(r, tag):
    for nm, v in r['rows'].items():
        print(f"[{tag}] C={r['C']:6.2f} D={r['D']:.2f} h={r['h']:.2f} kappa1={r['kappa1']:.4f} ts'={r['tsp']:.4f} {nm:9s}: "
              f"exact {v['exact']:+.5f} probe {v['probe']:+.5f} backreacted {v['backreacted']:+.5f} | C*total {v['C_total']:+.4f} = "
              f"classical {v['C_classical']:+.4f} + quantum {v['C_quantum']:+.4f} | k1*rel.quantum {v['k1_rel_quantum']:+.4f} |V| {v['absV_exact']:.5f}", flush=True)

if __name__ == "__main__":
    om, eps, ts = 0.2, 0.05, np.pi/4; out = {'fixed_bare': [], 'fixed_mu': [], 'h_scan': [], 'physical': []}
    # (1) the paper's bare protocol: Delta = 2, h = 1, tau_s = pi/4, C = 10, 20
    for C in (10., 20.):
        r = split(C, ts, 2., om, 1., eps); show(r, 'bare'); out['fixed_bare'].append(r)
    # (2) check of the backreaction mapping: fixed mu = Delta/C = 0.2 -> quantum part must converge (no O(mu C) remainder)
    for C, D in ((20., 4.), (40., 8.)):
        r = split(C, ts, D, om, 1., eps); show(r, 'mu=0.2'); out['fixed_mu'].append(r)
    # (3) signal dependence at C = 20, Delta = 2
    for h in (0.5, 1.5):
        r = split(20., ts, 2., om, h, eps); show(r, 'h-scan'); out['h_scan'].append(r)
    # (4) physical variables: S - S0 (with clock) = 2 pi k1, ts' = pi/4, m = Delta = 2
    for k1 in (10., 20.):
        Cb, tsb, x1 = B.phys_to_bare(k1, np.pi/4, 2.)
        r = split(Cb, tsb, 2., om, 1., eps); r['k1_target'] = k1; r['m_over_mmax'] = 2./B.m_max(k1, np.pi/4)
        show(r, f'phys k1={k1}'); out['physical'].append(r)
    json.dump(out, open('03_phase3_quantum/results/step3k_split.json', 'w'), indent=1, default=float)
