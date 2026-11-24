"""finite-L test: MC/planar - 1 at fixed k/L (=0.0533 and 1.78) for L = 450, 900, 1800 (thread regime, y=0.9)"""
import numpy as np
from planar import Planar
from analyze import ms
rows = [('thread@0.5', 'data/fs_thread_half.npz', (24, 800), 450), ('thread', 'data/main_thread.npz', (48, 1600), 900),
        ('thread@2', 'data/fs_thread_double.npz', (96, 3200), 1800)]
print('| L | k | N_samp | own MC/planar - 1 | unproj MC/planar - 1 | other MC/planar - 1 | rho00 |')
print('|---|---|---|---|---|---|---|')
for reg, fn, ks, L in rows:
    D = np.load(fn); yi = list(D['ys']).index(0.9); n = int(D['nsamp'])
    for k in ks:
        P = Planar(reg, k).predict(0.9, 0.05)
        un = D[f'unproj_rho10_odd_{k}'][:, yi]; ow = D[f'own_rho10_odd_{k}'][:, yi]; ot = un - ow
        out = []
        for x, p in ((ow, P['own']), (un, P['unproj']), (ot, P['other'])):
            r = (x / p).real          # projection on the predicted direction
            out.append(f'{r.mean()-1:+.4f} ± {r.std(ddof=1)/np.sqrt(n):.4f}')
        r00 = D[f'unproj_rho00_{k}'].real
        print(f'| {L} | {k} | {n} | ' + ' | '.join(out) + f' | {r00.mean():.5f} ± {r00.std(ddof=1)/np.sqrt(n):.5f} |')
