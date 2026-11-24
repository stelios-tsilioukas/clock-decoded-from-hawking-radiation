"""Step 3f: exact Ramsey visibility of the matter clock in physical variables (k1, tsp, clock mass D; pulses of width eps_phys).
Q = ln V_exact - ln V_cl,phys  (complex: Re = visibility, Im = phase), reported times k1 = (S - S0)/2 pi;
also Q(omega) - Q(0): the omega-dependent (clock) part."""
import numpy as np, sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3f_ramsey as R, step3d_backreaction as B
tsp = np.pi/4; dH = B.dy_horizon(tsp); y1, y2 = dH, 0.0
cases = eval(sys.argv[1]); tag = sys.argv[2]; epsp = 0.05
rows = []
def bare_cl(k1, D, om, h):
    C, ts, x1 = B.phys_to_bare(k1, tsp, D); return R.ramsey_cl(ts, y1/x1, y2/x1, epsp/x1, D, om, h)[0]
for k1, D, om, h in cases:
    t = time.time(); C, ts, x1 = B.phys_to_bare(k1, tsp, D)
    kmax = max(1.8*C + 8*np.sqrt(C) + 10, 3*np.sqrt(2*C*x1/epsp)); N = int(min(3200, max(1200, kmax/0.045)))
    X = R.ExactRamsey(C, kmax, N); out = {}
    for w in (om, 0.0):
        lv, _ = X.ramsey(ts, y1/x1, y2/x1, epsp/x1, D, w, h)
        lcp = R.ramsey_cl(tsp, y1, y2, epsp, D, w, h)[0]
        out[w] = (lv, lcp)
    Q = {w: k1*(out[w][0] - out[w][1]) for w in out}
    # one-loop prediction in physical variables: one loop (bare->phys at large k1) + backreaction of the map
    pred = {}
    for w in (om, 0.0):
        big = 1e4; br = big*(bare_cl(big, D, w, h) - R.ramsey_cl(tsp, y1, y2, epsp, D, w, h)[0])
        pred[w] = R.lnV_oneloop(tsp, y1, y2, epsp, D, w, h) + br
    row = dict(k1=k1, S_minus_S0=2*np.pi*k1, D=D, omega=om, h=h, m_over_mmax=D/B.m_max(k1, tsp), C_bare=C, N=N, kmax=kmax,
               lnV_cl_phys=[out[om][1].real, out[om][1].imag], lnV_exact=[out[om][0].real, out[om][0].imag],
               k1Q=[Q[om].real, Q[om].imag], k1Q0=[Q[0.0].real, Q[0.0].imag], k1Q_clock=[(Q[om]-Q[0.0]).real, (Q[om]-Q[0.0]).imag],
               pred=[pred[om].real, pred[om].imag], pred0=[pred[0.0].real, pred[0.0].imag])
    rows.append(row)
    print(f"S-S0={2*np.pi*k1:6.1f} D={D} w={om} h={h} (m/mmax {row['m_over_mmax']:.2f}, C={C:.2f}, N={N}, {time.time()-t:.0f}s): "
          f"V_cl={np.exp(out[om][1].real):.4f} | k1*Q = {Q[om].real:+.4f}{Q[om].imag:+.4f}i [1-loop {pred[om].real:+.4f}{pred[om].imag:+.4f}i] | "
          f"k1*Q(0) = {Q[0.0].real:+.4f}{Q[0.0].imag:+.4f}i [{pred[0.0].real:+.4f}{pred[0.0].imag:+.4f}i] | clock part {(Q[om]-Q[0.0]).real:+.4f}{(Q[om]-Q[0.0]).imag:+.4f}i", flush=True)
json.dump(dict(tsp=tsp, y1=y1, y2=y2, eps_phys=epsp, rows=rows), open(f'03_phase3_quantum/results/step3f_physical_{tag}.json', 'w'), indent=1)
