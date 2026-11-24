"""Tables from MC output + planar predictions.  usage: python analyze.py data/main_thread.npz [...]"""
import sys, numpy as np
from planar import Planar


def ms(x):
    x = np.asarray(x); n = len(x)
    return x.mean(), (x.real.std(ddof=1) + 1j * x.imag.std(ddof=1)) / np.sqrt(n) if n > 1 else 0j


def fmt(c, e=None, d=5):
    s = f'{c.real:+.{d}f}{c.imag:+.{d}f}i'
    if e is not None:
        s += f' (±{e.real:.{d}f}, ±{e.imag:.{d}f})'
    return s


def table(fn, yidx=0):
    D = np.load(fn)
    reg = str(D['reg']); ks = D['ks']; ys = D['ys']; n = int(D['nsamp']); eps = 0.05
    y = ys[yidx]
    print(f'\n### regime {reg}, y = {y}, N_samples = {n}  (eps-odd parts at eps = {eps}; errors = 1 s.e.m. on Re, Im)')
    rows = []
    for k in ks:
        un = D[f'unproj_rho10_odd_{k}'][:, yidx]; ow = D[f'own_rho10_odd_{k}'][:, yidx]
        ot = un - ow
        r00 = D[f'unproj_rho00_{k}'].real; r00o = D[f'own_rho00_{k}'].real
        ovu = D[f'unproj_ov_odd_{k}'][:, yidx]; ovo = D[f'own_ov_odd_{k}'][:, yidx]
        P = Planar(reg, k).predict(y, eps)
        rows.append((k, ms(un), ms(ow), ms(ot), ms(r00), ms(r00o), ms(ovu), ms(ovo), P))
    print('| k | rho10 unproj (MC) | planar | rho10 own (MC) | planar | other = unproj-own (MC) | planar |')
    print('|---|---|---|---|---|---|---|')
    for k, un, ow, ot, r00, r00o, ovu, ovo, P in rows:
        print(f'| {k} | {fmt(*un)} | {fmt(P["unproj"])} | {fmt(*ow)} | {fmt(P["own"])} | {fmt(*ot)} | {fmt(P["other"])} |')
    print('\n| k | rho00 (unproj model) | rho00 (own model) | <Psi1|Phi> unproj model (MC) | <Psi1|Phi> own model (MC) | planar |')
    print('|---|---|---|---|---|---|')
    for k, un, ow, ot, r00, r00o, ovu, ovo, P in rows:
        print(f'| {k} | {r00[0].real:.5f} ± {r00[1].real:.5f} | {r00o[0].real:.5f} ± {r00o[1].real:.5f} | {fmt(*ovu)} | {fmt(*ovo)} | {fmt(P["overlap"])} |')
    print('\n| k | |other/own| MC | arg(own) MC [deg] | arg(other) MC [deg] | |other/own| planar | arg(own) pl | arg(other) pl | own MC/planar - 1 | unproj MC/planar - 1 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for k, un, ow, ot, r00, r00o, ovu, ovo, P in rows:
        R = np.abs(ot[0] / ow[0]); Rp = np.abs(P['other'] / P['own'])
        print(f'| {k} | {R:.4f} | {np.degrees(np.angle(ow[0])):.2f} | {np.degrees(np.angle(ot[0])):.2f} | {Rp:.4f} | '
              f'{np.degrees(np.angle(P["own"])):.2f} | {np.degrees(np.angle(P["other"])):.2f} | '
              f'{abs(ow[0]/P["own"])-1:+.4f} | {abs(un[0]/P["unproj"])-1:+.4f} |')
    return D


def yscan(fn, k):
    D = np.load(fn); reg = str(D['reg']); ys = D['ys']; n = int(D['nsamp'])
    print(f'\n### y-scan, regime {reg}, k = {k}, N_samples = {n}')
    print('| y | own (MC) | other (MC) | |other/own| | arg own | arg other | own planar | other planar | <Psi1|Phi> planar |')
    print('|---|---|---|---|---|---|---|---|---|')
    order = np.argsort(ys)
    P = Planar(reg, k); I = P.integrals()
    for j in order:
        y = ys[j]
        un = D[f'unproj_rho10_odd_{k}'][:, j]; ow = D[f'own_rho10_odd_{k}'][:, j]
        o = ms(ow); t = ms(un - ow); r = P.predict(y, 0.05, I)
        print(f'| {y} | {fmt(*o)} | {fmt(*t)} | {abs(t[0]/o[0]):.4f} | {np.degrees(np.angle(o[0])):.2f} | '
              f'{np.degrees(np.angle(t[0])):.2f} | {fmt(r["own"])} | {fmt(r["other"])} | {fmt(r["overlap"])} |')


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        D = table(fn)
        if 'thread2' in fn:
            yscan(fn, 8000)
        else:
            yscan(fn, 8000)
