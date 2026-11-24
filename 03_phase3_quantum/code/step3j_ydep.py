"""Step 3j: y-dependence of the own (absorbed) and other-placement pieces (thread2 regime, k = 8000)."""
import numpy as np, sys, json
sys.path.insert(0, '03_phase3_quantum/code')
from step3j_absorption import RMTAbs, REGIMES
out = []
k = 8000
for y in [0.0, 0.45, 0.9, 1.35]:
    P = dict(REGIMES['thread2'], y=y); M = RMTAbs(P); eps = P['eps']
    rt, ra = [], []
    for smp in range(2):
        AA = M.draw2(np.random.default_rng(7000 + smp)); Q = M.wishart(np.random.default_rng(9000 + 97*smp + k), k)
        for nm, lst in (('true', rt), ('abs', ra)):
            o = M.decode(AA[nm], Q, eps); lst.append((o[+1]['r10'] - o[-1]['r10'])/2/eps)
    rt, ra = np.mean(rt), np.mean(ra); thr = M.planar(np.log(k))['r10']['V-row (threading)']/eps
    print(f"y={y:4.2f}  absorbed {ra:+.4f}  threading {thr:+.4f}  other {rt-ra:+.4f}  |other| {abs(rt-ra):.4f} arg(other)/pi {np.angle(rt-ra)/np.pi:+.3f}  arg(own)/pi {np.angle(ra)/np.pi:+.3f}", flush=True)
    out.append(dict(y=y, absorbed=[ra.real, ra.imag], threading=[thr.real, thr.imag], other=[(rt-ra).real, (rt-ra).imag]))
json.dump(out, open('03_phase3_quantum/results/step3j_ydep.json', 'w'), indent=1)
