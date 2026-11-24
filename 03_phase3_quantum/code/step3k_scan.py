"""Step 3k: omega, h, eps dependence of the quantum part of the stopwatch reading, physical variables
(S - S0 = 2 pi k1 with the clock, ts' = pi/4, m = Delta = 2), horizon label."""
import numpy as np, sys, json
sys.path.insert(0, '03_phase3_quantum/code')
import step3d_backreaction as B
from step3k_split import split
k1 = float(sys.argv[1]) if len(sys.argv) > 1 else 20.
Cb, tsb, x1 = B.phys_to_bare(k1, np.pi/4, 2.)
out = []
for om, h, eps in ((0.2, 1.0, 0.05), (0.1, 1.0, 0.05), (0.05, 1.0, 0.05), (0.1, 0.5, 0.05), (0.05, 0.25, 0.05), (0.2, 1.0, 0.025)):
    r = split(Cb, tsb, 2., om, h, eps, labels=('horizon',)); v = r['rows']['horizon']
    print(f"k1={k1} omega={om:.2f} h={h:.2f} eps={eps:.3f}: exact {v['exact']:+.6f} backreacted {v['backreacted']:+.6f} probe {v['probe']:+.6f} | k1*rel.quantum {v['k1_rel_quantum']:+.4f}  |V| {v['absV_exact']:.5f}  N={r['N']}", flush=True)
    out.append(dict(omega=om, h=h, eps=eps, **v, N=r['N']))
json.dump(out, open(f'03_phase3_quantum/results/step3k_scan_k{int(k1)}.json', 'w'), indent=1, default=float)
