#!/usr/bin/env python3
"""
Independent check (agent 'threading'): threading vs caps pairings of two clock lines
in the fully connected two-replica PSSY geometry (disk level, non-crossing chords).

Regions / weights (see report.md):
  caps     Z_C = int dmu(k) y(k)^2 e^{-s_b 4(pi-tau) k^2} [ H(k; 2 tau,      n=0) ]^2
  threading Z_T = int dmu(k)        e^{-s_b 4 tau k^2}    [ H(k; 2(pi-tau), n=1) ]^2
  H(k; l, n) = int dmu(a) y(a)^n e^{-s_b l a^2} g_D(a, k)
with dmu(k) = 2k sinh(2 pi k) dk, s_b = 1/(2C), g_D(a,b) = Gamma(D +- ia +- ib)/(2 pi^2 Gamma(2D)),
y(k) = |Gamma(mu-1/2+ik)|^2 / Gamma(mu-1/2)^2 (PSSY brane factor, normalised; mu=inf -> y=1).

Everything is done in log space with trapezoid sums (integrands are analytic, so the
trapezoid rule converges exponentially in 1/h).
"""
import numpy as np
from scipy.special import loggamma, gammaln, logsumexp
from scipy.optimize import brentq
import sys, json, time

PI = np.pi


def log_dmu(p):
    # log(2 p sinh(2 pi p)) = log p + 2 pi p + log(1 - e^{-4 pi p})
    return np.log(p) + 2 * PI * p + np.log1p(-np.exp(-4 * PI * p))


def log_y(p, mu):
    if mu is None or np.isinf(mu):
        return np.zeros_like(p)
    return 2 * np.real(loggamma(mu - 0.5 + 1j * p)) - 2 * gammaln(mu - 0.5)


def log_g_matrix(a, k, D, chunk=400):
    """log g_D(a_i, k_j) as a (len(a), len(k)) float array."""
    out = np.empty((len(a), len(k)))
    c0 = -gammaln(2 * D) - np.log(2 * PI ** 2)
    for s in range(0, len(a), chunk):
        A = a[s:s + chunk, None]
        out[s:s + chunk] = (2 * np.real(loggamma(D + 1j * (A + k[None, :])))
                            + 2 * np.real(loggamma(D + 1j * (A - k[None, :]))) + c0)
    return out


def kappa_star(lam, n=2):
    """classical saddle k*/C of the n-replica connected geometry with brane factor
    mu = lam*C (lam=inf -> 1/n):  2 pi - 2 pi n kappa - 2 n atan(kappa/lam) = 0."""
    if lam is None or np.isinf(lam):
        return 1.0 / n
    f = lambda x: 2 * PI - 2 * PI * n * x - 2 * n * np.arctan(x / lam)
    return brentq(f, 1e-12, 1.0 / n)


def probe_limit(D, tau, lam=None):
    ks = kappa_star(lam, 2)
    return 4 * D * np.log(np.tan(ks * tau))


class Setup:
    def __init__(self, C, D, mu=None, h=0.05, W=None, margin=120.0):
        self.C, self.D, self.mu, self.h = C, D, mu, h
        sb = 1.0 / (2 * C)
        self.sb = sb
        lam = None if (mu is None or np.isinf(mu)) else mu / C
        kc = kappa_star(lam, 2) * C
        if W is None:
            W = max(25.0, 6 * np.sqrt(C))
        klo = max(h, kc - W)
        self.k = np.arange(klo, kc + W, h)
        self.a = np.arange(h, kc + W + margin, h)
        self.lg = log_g_matrix(self.a, self.k, D)
        self.ldmu_a, self.ldmu_k = log_dmu(self.a), log_dmu(self.k)
        self.ly_a, self.ly_k = log_y(self.a, mu), log_y(self.k, mu)
        self.lw = np.log(h)

    def logH(self, ell, n):
        la = self.ldmu_a + n * self.ly_a - self.sb * ell * self.a ** 2 + self.lw
        return logsumexp(la[:, None] + self.lg, axis=0)

    def logZ_caps(self, tau):
        lH = self.logH(2 * tau, 0)
        lk = self.ldmu_k + 2 * self.ly_k - self.sb * 4 * (PI - tau) * self.k ** 2 + self.lw
        return logsumexp(lk + 2 * lH)

    def logZ_thread(self, tau):
        lH = self.logH(2 * (PI - tau), 1)
        lk = self.ldmu_k - self.sb * 4 * tau * self.k ** 2 + self.lw
        return logsumexp(lk + 2 * lH)

    def edge_check(self, tau):
        """fraction of log-weight at the ends of the k grid (truncation check)"""
        lH = self.logH(2 * tau, 0)
        lk = self.ldmu_k + 2 * self.ly_k - self.sb * 4 * (PI - tau) * self.k ** 2 + 2 * lH
        return max(lk[0], lk[-1]) - lk.max(), self.k[np.argmax(lk)]


def run(Cs, Ds, taus_over_pi, mu_of_C=lambda C: None, h=0.05, W=None, margin=120.0):
    rows = []
    for C in Cs:
        mu = mu_of_C(C)
        for D in Ds:
            t0 = time.time()
            S = Setup(C, D, mu=mu, h=h, W=W, margin=margin)
            for tp in taus_over_pi:
                tau = tp * PI
                lt, lc = S.logZ_thread(tau), S.logZ_caps(tau)
                lam = None if mu is None else mu / C
                rows.append(dict(C=C, D=D, mu=mu, tau_over_pi=tp, logT=lt, logC=lc,
                                 logratio=lt - lc, probe=probe_limit(D, tau, lam),
                                 edge=S.edge_check(tau)[0]))
            print(f"# C={C} D={D} mu={mu}: {time.time()-t0:.1f}s  grid a:{len(S.a)} k:{len(S.k)}",
                  file=sys.stderr, flush=True)
    return rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    taus = [0.25, 0.375, 0.5, 0.625, 0.75]
    Ds = [0.5, 1, 2, 4]
    if mode == "main":
        rows = run([10, 40], Ds, taus)
    elif mode == "C160":
        rows = run([160], Ds, taus, h=0.1)
    elif mode == "conv":
        # grid convergence: step h, k-window W and a-margin
        rows = []
        for (h, W, m) in [(0.1, None, 120.0), (0.05, None, 120.0), (0.025, None, 120.0),
                          (0.05, 40.0, 200.0)]:
            r = run([10, 40], [0.5, 4], [0.25, 0.75], h=h, W=W, margin=m)
            for x in r:
                x.update(h=h, W=W, margin=m)
            rows += r
    elif mode == "finite_mu":
        # PSSY brane factor with mu = lam*C, lam = 1 (finite tension check of generalised probe formula)
        rows = run([10, 40, 160], [1], taus + [0.65], mu_of_C=lambda C: 1.0 * C, h=0.05)
    out = f"rederivation/threading/results_{mode}.json"
    json.dump(rows, open(out, "w"), indent=1)
    for r in rows:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
