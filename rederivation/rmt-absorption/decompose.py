"""Planar decomposition of own / other placement over (alpha = final B energy, beta = energy between signal and
clock line).  At y=0 every weight is real and >=0 (own) or <=0 (other); y enters only as exp(i y (E_alpha - E_beta))."""
import numpy as np, sys
from planar import Planar
np.set_printoptions(precision=4, suppress=True, linewidth=160)
for reg in ('thread', 'thread2'):
    for k in (48, 8000):
        P = Planar(reg, k); I = P.integrals()
        w0 = P.wtensor(0.0).real; nrm = 0.05 / 2 / np.sqrt(P.N[0] * P.N[1])
        own = nrm * np.einsum('abg,a,ag->ab', w0, I['m'], I['T'])
        unp = nrm * np.einsum('abg,a,abg->ab', w0, I['m'], I['Tu'])
        oth = unp - own
        ov = 0.05 * P.kk / np.sqrt(P.N[0] * P.N[1]) * w0.sum(2)
        dE = P.S.Eb[:, None] - P.S.Eb[None, :]
        print(f'== {reg} k={k}')
        for name, c in (('overlap', ov), ('own', own), ('other', oth)):
            print(f' {name:8s} total(y=0) {c.sum():+.6f}  frac alpha=0: {c[0].sum()/c.sum():.4f}  frac (alpha,beta)=(0,0): {c[0,0]/c.sum():.4f}'
                  f'  <E_beta-E_alpha> = {np.sum(-dE*c)/c.sum():.4f}  beta-distribution {np.round(c.sum(0)/c.sum(),4)}')
        print('  own/overlap ratio per beta (alpha summed):', np.round(own.sum(0) / ov.sum(0), 4))
        print('  other/own ratio per beta:', np.round(oth.sum(0) / own.sum(0), 4))
