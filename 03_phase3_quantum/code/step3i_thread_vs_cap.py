"""
Step 3i: which planar pairing of the clock lines dominates the connected replica geometries in JT.
Two-replica connected weights (same clock level):
  threading: clock lines join bra_1 -> ket_2 and bra_2 -> ket_1; shared region = central (middle arcs [u+,u-]);
      Z2_thr = Int dmu(s) e^{-2 t s^2} [Int dmu(E) e^{-t0 E^2} g_D(s,E)]^2
  caps:      each clock line returns to its own boundary; shared region = brane side;
      Z2_cap = Int dmu(E) e^{-2 t0 E^2} [Int dmu(s) e^{-t s^2} g_D(s,E)]^2
  with t = 2 s_b tau_s (middle arc), t0 = beta - t (brane side).  Both are planar disk geometries of the same order in e^{S0};
  only threading carries the code index between replicas (needed for Petz decoding from R).
Physical reading: caps dominate when the clock sits outside the horizon on the time-symmetric slice (tau_s < pi/2:
B's entanglement wedge); threading dominates when it sits behind it (tau_s > pi/2: the island).
"""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
from step3h_replica import Replica, lse
out = []
for C in (10.0, 40.0):
    for D in (0.5, 1.0, 2.0, 4.0):
        row = []
        for ts in np.pi*np.array([0.25, 0.375, 0.5, 0.625, 0.75]):
            R = Replica(C, ts, D, N=1600); t = 2*R.s*ts; t0 = R.t0; base = R.lmu + R.lwq
            inb, inc = R._inner(D, t0).real, R._inner(D, t).real
            d = lse(base - 2*t*R.k**2 + 2*inb) - lse(base - 2*t0*R.k**2 + 2*inc)
            row.append(float(d))
        out.append(dict(C=C, Delta=D, tau_s_over_pi=[0.25, 0.375, 0.5, 0.625, 0.75], log_thread_over_cap=row))
        print(f"C={C:4.0f} Delta={D}: log(Z2_thread/Z2_cap) at tau_s/pi = 0.25..0.75: " + "  ".join(f"{v:+7.3f}" for v in row)
              + f"   | 4 Delta log tan(tau_s/2) at 0.25 pi: {4*D*np.log(np.tan(np.pi/8)):+.3f}", flush=True)
json.dump(out, open('03_phase3_quantum/results/step3i_thread_vs_cap.json', 'w'), indent=1)
