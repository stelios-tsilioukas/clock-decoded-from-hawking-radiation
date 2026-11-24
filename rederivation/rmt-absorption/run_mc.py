"""Monte-Carlo driver.  usage: python run_mc.py REGIME NSAMP SEED k1,k2,... y1,y2,... OUT.npz [fpow]"""
import sys, time
import numpy as np
from model import Setup, petz_quantities


def one_sample(S, rng, ks, ys):
    eps = S.p['eps']
    m = S.draw_matter(rng)
    X0 = S.Xmat(m['O'])
    XZ = S.Xmat(m['Z'])
    Wu = S.Xmat(S.W_of(m['V'], m['O']))    # e^{-tc H/2} W e^{-tb H/2} F, unprojected (W built with O0)
    Wo = S.Xmat(S.W_of(m['V'], m['Ot']))   # own-placement model (W built with independent copy Ot)
    Yu = [S.Ymat(m['V'], m['O'], y) for y in ys]
    Yo = [S.Ymat(m['V'], m['Ot'], y) for y in ys]
    res = {}
    for k in ks:
        T = S.brane(rng, k)
        Psi0 = X0 @ T; N0 = np.vdot(Psi0, Psi0).real; Psi0 /= np.sqrt(N0)
        PZ = XZ @ T
        r = {}
        for model, Wx, Ys in (('unproj', Wu, Yu), ('own', Wo, Yo)):
            PW = Wx @ T
            Phis = [Yx @ T / np.sqrt(N0) for Yx in Ys]
            vals = {}
            for sgn in (+1, -1):
                Psi1 = PZ + sgn * eps * PW
                Psi1 = Psi1 / np.sqrt(np.vdot(Psi1, Psi1).real)
                rho10, rho00 = petz_quantities(Psi0, Psi1, Phis)
                ov = [np.vdot(Psi1, Ph) for Ph in Phis]        # <Psi1|Phi>
                vals[sgn] = (np.array(rho10), rho00, np.array(ov))
            r[model] = dict(
                rho10_odd=(vals[1][0] - vals[-1][0]) / 2,
                rho10_even=(vals[1][0] + vals[-1][0]) / 2,
                rho00=(vals[1][1] + vals[-1][1]) / 2,
                ov_odd=(vals[1][2] - vals[-1][2]) / 2,
                ov_even=(vals[1][2] + vals[-1][2]) / 2)
        res[k] = r
    return res


def main():
    reg = sys.argv[1]; ns = int(sys.argv[2]); seed = int(sys.argv[3])
    ks = [int(x) for x in sys.argv[4].split(',')]
    ys = [float(x) for x in sys.argv[5].split(',')]
    out = sys.argv[6]
    fpow = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    S = Setup(reg, fpow=fpow)
    rng = np.random.default_rng(seed)
    keys = ['rho10_odd', 'rho10_even', 'rho00', 'ov_odd', 'ov_even']
    data = {f'{mo}_{kk}_{k}': [] for mo in ('unproj', 'own') for kk in keys for k in ks}
    t0 = time.time()
    for s in range(ns):
        res = one_sample(S, rng, ks, ys)
        for k in ks:
            for mo in ('unproj', 'own'):
                for kk in keys:
                    data[f'{mo}_{kk}_{k}'].append(res[k][mo][kk])
        np.savez(out, ks=ks, ys=ys, reg=reg, fpow=fpow, nsamp=s + 1,
                 **{kk: np.array(v) for kk, v in data.items()})
        print(f'sample {s+1}/{ns} done, {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
