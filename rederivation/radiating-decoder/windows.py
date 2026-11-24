"""Find integration windows (support of each momentum variable) from magnitude marginals on coarse grids."""
import numpy as np, sys, json
from core import *

def win(x, l, cut=35., pad=2.0):
    l = l.real - l.real.max(); idx = np.where(l > -cut)[0]
    lo, hi = x[idx[0]] - pad, x[idx[-1]] + pad
    edge = (l[0], l[-1])
    return [max(lo, 0.05), hi], x[np.argmax(l)], edge

def find(k1, ts_phys=np.pi/4, m=2., Delta=2., De=1., ee=0.0125, eps=0.05, hV=1.0, h=0.5, scale=None):
    p = physical_map(k1, ts_phys, m)
    sc = k1/20. if scale is None else scale
    g = dict(A=grid(0, 300*sc, h), I=grid(0, 300*sc, h), D=grid(0, 120*sc, h), B=grid(0, 500*sc, h),
             E=grid(0, 700*sc, h), S=grid(0, 300*sc, h))
    s = Setup(p['C'], p['ts'], Delta=Delta, De=De, ee=ee, eps=eps, hV=hV, grids=g)
    A, _ = g['A']; I, _ = g['I']; D, _ = g['D']; B, _ = g['B']; E, _ = g['E']; S, _ = g['S']
    res = {}
    lD = logmu(D) - s.tD*D**2
    for a in (0, 1):
        w = s.weights(a)
        lc = s.logcap(a, 0., I) + logmu(I)
        L1 = logg(w['psichi'], D[:, None], I[None, :]); L2 = logg(w['Vchi'], I[:, None], B[None, :])
        LI = s.logItri(a, 0., D, B)
        lB = logmu(B) + s.logWe(B, True)
        base = lD[:, None] + lB[None, :] - logg(s.Dchi, D[:, None], B[None, :]) + LI.real  # times conj ket-other
        T = base + LI.real
        res.setdefault('D', []).append(win(D, lse(T, 1))); res.setdefault('B', []).append(win(B, lse(T, 0)))
        # I marginal: sum_{D,B} exp(base[D,B] + L1[D,I] + lc[I] + L2[I,B])
        lIm = np.array([lse((base + L1[:, i][:, None] + L2[i][None, :]).ravel(), 0) for i in range(len(I))]) + lc
        res.setdefault('I', []).append(win(I, lIm))
        rest = lIm - s.logcap(a, 0., I).real
        lA = logmu(A) - s.sb*(s.ts - s.eps/2)*A**2
        TA = lA[:, None] + logg(w['psiV'], A[:, None], I[None, :]) + rest[None, :] - lc[None, :] + logmu(I)[None, :]
        res.setdefault('A', []).append(win(A, lse(TA, 1)))
        # decoded with emission: sigma and E, D-in-S
        lge = logg(s.De, S[:, None], E[None, :])
        lFc = s.logF_chi(S)
        lP = lse((logmu(S) - s.sb*(2*s.ts - 2*s.ee)*S**2 + lFc)[:, None] + lge, 0)
        lE = logmu(E) - 2*s.sb*s.ee*E**2 - lP
        LIS = s.logItri(a, 0., D, S)
        lSa = lse(lD[:, None] + LIS, 0).real
        lws = logmu(S) - s.sb*(s.ts + s.eps/2 - 2*s.ee)*S**2
        LK = lse((lws + lSa)[:, None] + lge, 0).real
        res.setdefault('E', []).append(win(E, lE + 2*LK))
        Ts = (lws + lSa)[:, None] + lge + (lE + LK)[None, :]
        res.setdefault('S', []).append(win(S, lse(Ts, 1)))
        # reference sigma in P
        TP = (logmu(S) - s.sb*(2*s.ts - 2*s.ee)*S**2 + lFc)[:, None] + lge + (lE + 2*LK)[None, :] - lP[None, :]
        res.setdefault('S', []).append(win(S, lse(TP, 1)))
        # D inside S_a(sigma) weighted by sigma marginal
        sm = lse(Ts, 1) - lSa
        TD = lD[:, None] + LIS.real + sm[None, :]
        res.setdefault('D', []).append(win(D, lse(TD, 1)))
        # D in F_chi(sigma) weighted
        TDf = lD[:, None] + logg(s.Dchi, D[:, None], S[None, :]) + (lse(TP, 1) - lFc)[None, :]
        res.setdefault('D', []).append(win(D, lse(TDf, 1)))
        # Y0 s-window
        lS0 = logmu(B) - s.logF_chi(B) - s.sb*s.eps*B**2
        LS0 = lse(lD[:, None] + LI, 0).real
        res.setdefault('B', []).append(win(B, lS0 + 2*LS0))
    out = {}
    for k, v in res.items():
        lo = min(r[0][0] for r in v); hi = max(r[0][1] for r in v)
        out[k] = [lo, hi]
        print(k, [(round(r[0][0], 1), round(r[0][1], 1), round(r[1], 1), [round(e, 0) for e in r[2]]) for r in v])
    return out

if __name__ == '__main__':
    k1 = float(sys.argv[1]); De = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    ee = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0125
    out = find(k1, De=De, ee=ee)
    print(json.dumps(out))
    json.dump(out, open(f'win_k{int(k1)}_De{De}_ee{ee}.json', 'w'))
