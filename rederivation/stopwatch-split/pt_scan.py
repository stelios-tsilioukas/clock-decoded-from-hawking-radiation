"""One-loop coefficient a(omega,h) at the question-4 point; checks (non)linearity in omega, h."""
import numpy as np, json
from sw import phys_to_bare, yH_of, R_cl, bare_to_phys
from oneloop import a_coeff
k1, taup, m = 20.0, np.pi/4, 2.0
d = phys_to_bare(k1, taup, m); C, k, tau = d['C'], d['kappa1'], d['tau']
lab = yH_of(taup); y = lab/k
res = {}
for eps in [0.05/k, 1e-4]:
    for om in [1e-4, 0.05, 0.1, 0.2]:
        for h in [1e-4, 0.25, 0.5, 1.0]:
            a = a_coeff(tau, y, eps, m, om, h)
            res[f'{eps:.4g}_{om}_{h}'] = a
            print(f'eps={eps:.4g} om={om} h={h} a={a:.6f}', flush=True)
# backreaction slope at this kinematics (C->inf, fixed Delta): lim C*(R_cl(kap.)-R_cl(1.)) / Delta
Cb = 1e5
db = bare_to_phys(Cb, tau, 1.0); kb = db['kappa1']
for eps in [0.05/k, 1e-4]:
    b = Cb*(R_cl(kb*tau, kb*y, kb*eps) - R_cl(tau, y, eps))
    res[f'backreaction_slope_{eps:.4g}'] = b
    print('backreaction slope per unit Delta', eps, b)
json.dump(res, open('pt_scan.json','w'), indent=1)
