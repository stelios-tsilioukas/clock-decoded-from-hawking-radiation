"""
A2: exact disk-level (Schwarzian / JT, non-crossing chords) evaluation of
   M_p(y) = < B(+,y)^{-p/2}  B(y,-)^{-p/2}  B(+,-)^{-(m-p/2)} >
and of the tick-rate moments <R^p>_m = M_p / M_0, R = [B(+,-)/(B(+,y) B(y,-))]^{1/2}.

Disk rules (u-units, beta = 2pi, s = 1/(2C)):
   regions:  1: arc (-ts -> iy), length l1 = ts + i y, momentum k1
             2: arc (iy -> ts),  length l2 = ts - i y, momentum k2
             3: arc (ts -> 2pi - ts), length l3 = 2pi - 2ts, momentum k3
             inner triangle (no boundary arc), momentum q
   M_p = int dmu(q) F_{p/2}(q; l1) F_{p/2}(q; l2) F_{m-p/2}(q; l3)
   F_w(q; l) = int dmu(k) exp(-s l k^2) g_w(k, q)
   dmu(k) = 2k sinh(2 pi k) dk,  g_w(a,b) = Gamma(w +- ia +- ib) / (2 pi^2 Gamma(2w) (2C)^{2w})
The (2C)^{-2w} makes  <B^{-w}> -> B_cl^{-w}  at large C (derived in report), so R has classical value
R_cl = sin ts / (cosh y - cos ts).
M_0 = int dmu(q) exp(-s (l1+l2) q^2) F_m(q; l3)   (w -> 0 chords are delta functions).

All momentum integrals: trapezoid on k_j = j h (integrands even in k -> spectral accuracy).
"""
import numpy as np
from scipy.special import loggamma, gammaln
import sys, time, json

class Disk:
    def __init__(self, C, h=0.04, extra=100.0, nsig=15.0):
        self.C = C
        self.s = 1.0 / (2 * C)
        sig = np.sqrt(C / (2 * np.pi))
        K = C + nsig * sig + extra
        self.N = int(np.ceil(K / h))
        self.h = h
        self.k = h * np.arange(1, self.N + 1)
        k = self.k
        self.logrho = np.log(2 * k) + 2 * np.pi * k + np.log1p(-np.exp(-4 * np.pi * k)) - np.log(2)
        self._lg = {}

    def lgtab(self, w):
        """2 Re loggamma(w + i n h), n = 0..2N+2, minus normalisation"""
        if w not in self._lg:
            n = np.arange(0, 2 * self.N + 3)
            t = 2 * np.real(loggamma(w + 1j * n * self.h))
            self._lg[w] = (t, -np.log(2 * np.pi ** 2) - gammaln(2 * w) - 2 * w * np.log(2 * self.C))
        return self._lg[w]

    def logF(self, w, l, chunk=400):
        """complex log of F_w(q_i; l) on the grid q_i = k_i"""
        N, h, s, k = self.N, self.h, self.s, self.k
        t, c0 = self.lgtab(w)
        base = self.logrho - s * np.real(l) * k ** 2
        phase = np.exp(-1j * s * np.imag(l) * k ** 2)
        out = np.empty(N, dtype=complex)
        j = np.arange(1, N + 1)
        for a in range(0, N, chunk):
            i = np.arange(a + 1, min(a + chunk, N) + 1)
            L = base[None, :] + t[i[:, None] + j[None, :]] + t[np.abs(i[:, None] - j[None, :])] + c0
            M = L.max(axis=1)
            S = (np.exp(L - M[:, None]) * phase[None, :]).sum(axis=1) * h
            out[i - 1] = M + np.log(S)
        return out

    def outer(self, logterms):
        L = self.logrho + logterms
        M = np.max(np.real(L))
        return M + np.log(np.sum(np.exp(L - M)) * self.h)

def moments(C, ts, ys, ms, h=0.04, extra=100.0, nsig=15.0, pmax=2):
    D = Disk(C, h, extra, nsig)
    s, k = D.s, D.k
    l3 = 2 * np.pi - 2 * ts
    res = {}
    F3 = {}
    for m in ms:
        for p in range(0, pmax + 1):
            w = m - p / 2
            if w > 0 and w not in F3:
                F3[w] = D.logF(w, l3)
    F1 = {}
    for y in ys:
        for p in range(1, pmax + 1):
            F1[(y, p)] = D.logF(p / 2, ts + 1j * y)
    for m in ms:
        logM0 = D.outer(-s * 2 * ts * k ** 2 + F3[m])
        for y in ys:
            Rcl = np.sin(ts) / (np.cosh(y) - np.cos(ts))
            mom = {}
            for p in range(1, pmax + 1):
                w = m - p / 2
                if w <= 0:
                    continue
                f1 = F1[(y, p)]
                f2 = np.conj(f1)   # l2 = conj(l1), g real  ->  F(q; l2) = conj F(q; l1)
                logMp = D.outer(f1 + f2 + F3[w])
                mom[p] = np.exp(logMp - logM0)
            r = {"Rcl": Rcl}
            if 1 in mom:
                r["R1"] = mom[1].real; r["R1imag"] = mom[1].imag
                r["a"] = C * (mom[1].real / Rcl - 1)
            if 2 in mom:
                r["R2"] = mom[2].real
                r["v"] = C * (mom[2].real / mom[1].real ** 2 - 1)
            res[(m, y)] = r
    return res, D.N

def twopoint(C, ts, ms, h=0.04, extra=100.0):
    """<B(+,-)^{-m}> / B_cl^{-m} with B_cl = 4 sin^2 ts : checks the (2C)^{2w} normalisation."""
    D = Disk(C, h, extra)
    s, k = D.s, D.k
    logZ = D.outer(-s * 2 * np.pi * k ** 2 + 0j)
    out = {}
    for m in ms:
        logM0 = D.outer(-s * 2 * ts * k ** 2 + D.logF(m, 2 * np.pi - 2 * ts))
        out[m] = (np.exp(logM0 - logZ) * (4 * np.sin(ts) ** 2) ** m).real
    return out

if __name__ == "__main__":
    ts = np.pi / 4
    yH = np.log(np.cos(ts))
    ys = [yH, 0.0, -0.9 * yH, yH - 1, yH - 2]
    names = ["y_H", "0", "-0.9y_H", "y_H-1", "y_H-2"]
    Cs = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [20.0]
    ms = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3.0]
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04
    extra = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
    tag = sys.argv[5] if len(sys.argv) > 5 else "run"
    allres = []
    for C in Cs:
        t0 = time.time()
        res, N = moments(C, ts, ys, ms, h=h, extra=extra)
        for (m, y), r in res.items():
            nm = names[ys.index(y)]
            row = dict(C=C, m=m, y=y, label=nm, h=h, extra=extra, N=N, **{kk: float(vv) for kk, vv in r.items()})
            allres.append(row)
            print(f"C={C:6.1f} m={m:5.2f} {nm:8s} y={y: .5f} Rcl={r['Rcl']:.6f} <R>={r.get('R1', np.nan):.10f} "
                  f"Im={r.get('R1imag', np.nan):.1e} a={r.get('a', np.nan):.8f} v={r.get('v', np.nan):.8f}", flush=True)
        print(f"# C={C} N={N} time {time.time() - t0:.1f}s", flush=True)
    with open(f"exact_{tag}.json", "w") as fh:
        json.dump(allres, fh, indent=1)
