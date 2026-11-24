"""
Step 3i (open point 1) -- exact planar resolvent of the code-averaged radiation state sigma_R, and the exact
Petz lambda-integrals that replace PSSY's canonical approximation (3.44).

Ensemble: blocks j (grid points of the shared energy s) with rho_j states (rho-bar, includes d_code e^{S0}) and, for
each code level a = 0..d-1, a normalised energy distribution p_a(j).  Planar Schwinger-Dyson equation for
R(lambda) = Tr (lambda - sigma_R)^{-1} (PSSY 2.34, generalised to level-dependent weights; all boundaries of one
connected component carry the same level, because the level rides on the line that threads them):

    lambda R = k + sum_{a,j} (rho_j/d) w_aj R/(k - w_aj R),        w_aj = p_a(j)/rho_j .

Scaled variables Lambda = k lambda, gamma = R/k^2:
    Lambda gamma = 1 + sum_{a,j} P_aj gamma/(1 - W_aj gamma),   P_aj = p_a(j)/d,  W_aj = k w_aj.

Petz lambda-integrals (PSSY 3.40 with n -> -1/2; contour around the cut on the positive axis, counter-clockwise):
    J_a(j) = (1/2 pi i) oint dlambda lambda^{-1/2} R/(k - w_aj R) = k^{1/2} (1/2 pi i) oint dLambda Lambda^{-1/2} gamma/(1 - W_aj gamma)
Canonical approximation (PSSY 2.46, 3.44):  J_a(j) -> (lambda_0 + w_aj)^{-1/2}.

Method: the contour is an ellipse in z = log Lambda (|Im z| < pi keeps clear of the branch cut of Lambda^{-1/2}),
enclosing log of the whole spectrum.  gamma is found by Newton continuation from large real Lambda (where
gamma = 1/Lambda + ...), which keeps it on the physical sheet; the lower half follows by conjugation.  The periodic
trapezoid rule is spectrally accurate.  Checks: (1/2 pi i) oint dLambda gamma = 1 (k eigenvalues), (1/2 pi i) oint
dLambda Lambda gamma = 1 (Tr sigma_R = 1); entropy S = log k - (1/2 pi i) oint dLambda Lambda log(Lambda) gamma.
"""
import numpy as np

def lse(L, axis=None):
    L = np.asarray(L); o = np.max(np.real(L), axis=axis, keepdims=True)
    o = np.where(np.isfinite(o), o, 0.0)                         # all -inf slice -> -inf, not nan (audit M3)
    r = o + np.log(np.sum(np.exp(L - o), axis=axis, keepdims=True))
    return np.squeeze(r, axis=axis) if axis is not None else r.item()

class ExactResolvent:
    def __init__(self, lrho, lps, logk, M=4096, B=2.5, mL=6.0, mR=3.0, floor=-600.0):
        lps = [np.asarray(lp, float) for lp in lps]; d = len(lps); self.d = d; self.logk = logk
        lrho = np.asarray(lrho, float); self.lrho = lrho
        lP = np.array([lp - np.log(d) for lp in lps]); lW = np.array([logk + lp - lrho for lp in lps])
        keep = np.isfinite(lP) & (lP > floor)                      # drop blocks with negligible weight
        self.lP, self.lW, self.keep = lP, lW, keep
        self.P = np.where(keep, np.exp(np.where(keep, lP, 0.0)), 0.0)
        self.W = np.where(keep, np.exp(np.clip(lW, -700, 700)), 0.0)
        # canonical edge data (PSSY 2.39, 2.43) for the code-averaged ensemble
        lPbar = lse(lP, axis=0)                                   # log sum_a P_aj
        cum = np.logaddexp.accumulate(lrho); i = int(np.searchsorted(cum, logk)); self.ik = i
        self.lLam0 = lse(lPbar[i:]) if i < lrho.size else -np.inf  # log (k lambda_0)
        Wk = np.max(self.W[:, min(i, lrho.size - 1)])              # W at s_k
        Wmax = np.max(np.where(keep, self.W, 0.0))
        lo = np.logaddexp(self.lLam0, np.log(Wk) if Wk > 0 else -np.inf)
        hi = np.logaddexp(self.lLam0, np.log(Wmax))
        self.zL, self.zR = lo - mL, max(hi, 0.0) + mR
        self.c, self.A, self.B = 0.5*(self.zL + self.zR), 0.5*(self.zR - self.zL), B
        self.M = M; self._solve()

    # ---------------------------------------------------------------- Schwinger-Dyson on the contour
    def _F(self, g, Lam):
        r = 1.0/(1.0 - self.W*g)                    # 1/u; avoids complex overflow of u**2 when |W g| > 1e154 (audit M3)
        t = (self.P*g*r).sum(); dt = (self.P*r*r).sum()
        return Lam*g - 1.0 - t, Lam - dt

    def _newton(self, g, Lam, tol=1e-14, it=60):
        for _ in range(it):
            F, dF = self._F(g, Lam); step = F/dF; g = g - step
            if abs(step) < tol*abs(g): return g, True
        return g, False

    def _solve(self):
        M = self.M; phis = np.pi*np.arange(M//2 + 1)/(M//2)          # upper half, phi in [0, pi]
        z = self.c + self.A*np.cos(phis) + 1j*self.B*np.sin(phis)
        dz = -self.A*np.sin(phis) + 1j*self.B*np.cos(phis)
        g = np.empty(z.size, complex); Lam0 = np.exp(z[0])
        g0, ok = self._newton(1.0/Lam0 + 0j, Lam0); assert ok, "Newton failed at the starting point"
        g[0] = g0; prev, prevprev = g0, g0; phi_prev = 0.0
        for m in range(1, z.size):
            # adaptive sub-stepping between phi_{m-1} and phi_m
            nsub = 1
            while True:
                gg, good = prev, True; pp = prevprev
                for q in range(1, nsub + 1):
                    ph = phi_prev + (phis[m] - phi_prev)*q/nsub
                    zz = self.c + self.A*np.cos(ph) + 1j*self.B*np.sin(ph)
                    guess = 2*gg - pp if q > 1 or m > 1 else gg
                    gn, ok = self._newton(guess, np.exp(zz))
                    if not ok or abs(gn - gg) > 0.5*abs(gg) + 1e-300: good = False; break
                    pp, gg = gg, gn
                if good: break
                nsub *= 2
                if nsub > 4096: raise RuntimeError("continuation failed")
            prevprev, prev = (prev if nsub == 1 else pp), gg; g[m] = gg; phi_prev = phis[m]
        self.z, self.dz, self.g = z, dz, g
        self.wts = np.full(z.size, 1.0); self.wts[0] = self.wts[-1] = 0.5
        self.dphi = phis[1] - phis[0]

    def _oint(self, f):
        """(1/2 pi i) oint dz f(z), f evaluated on the upper half, f(z*) = f(z)* (real-analytic integrands)."""
        return (self.dphi/np.pi)*np.sum(self.wts*np.imag(f*self.dz))

    # ---------------------------------------------------------------- outputs
    def sum_rules(self):
        """(count, normalisation): (1/2pi i) oint dLambda gamma = (number of non-zero eigenvalues)/k and
        (1/2pi i) oint dLambda Lambda gamma = Tr sigma_R = 1.  The second suffers cancellation of O(e^{zR}) terms
        and is informative only when the spectrum spans a modest range (it is checked on the PSSY ensemble)."""
        Lam = np.exp(self.z); g = self.g
        return self._oint(Lam*g), self._oint(Lam**2*g)

    def entropy(self):
        Lam = np.exp(self.z)
        return self.logk - self._oint(Lam**2*self.z*self.g)

    def logJ(self, chunk=512):
        """log J_a(j) (exact, in units where J = (1/2pi i) oint dlambda lambda^{-1/2} R/(k - wR)); shape (d, N)."""
        Lam = np.exp(self.z); g = self.g; d, N = self.W.shape
        base = np.exp(0.5*self.z)*g*self.dz*self.wts                # Lambda^{1/2} gamma dz
        out = np.full((d, N), -np.inf)
        for a in range(d):
            for i0 in range(0, N, chunk):
                Wc = self.W[a, i0:i0+chunk]
                val = (self.dphi/np.pi)*np.imag(base[:, None]/(1.0 - Wc[None, :]*g[:, None])).sum(axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    out[a, i0:i0+chunk] = np.log(val) + 0.5*self.logk   # J = k^{1/2} * contour integral
        return out

    def logJ_canonical(self):
        """log (lambda_0 + w_aj)^{-1/2}."""
        lw = self.lW - self.logk                                    # log w_aj
        ll0 = self.lLam0 - self.logk
        return -0.5*np.logaddexp(ll0, lw)

# ======================================================================== validation against PSSY
def pssy_blocks(beta, S0=0.0, smax=12.0, N=3000):
    """PSSY section 2 model, large mu: rho(s) = s sinh(2 pi s)/(2 pi^2), y(s) = e^{-beta s^2/2}."""
    s = np.linspace(1e-6, smax, N); w = np.full(N, s[1] - s[0]); w[0] *= .5; w[-1] *= .5
    lrho = S0 + np.log(s) + np.log(np.sinh(2*np.pi*s)) - np.log(2*np.pi**2) + np.log(w)
    ly = lrho - beta*s**2/2; lp = ly - lse(ly)
    Sbh = -(np.exp(lp)*(lp - lrho)).sum()                          # thermal entropy (incl. S0)
    return s, lrho, lp, Sbh

if __name__ == "__main__":
    import json, sys
    out = {}
    # (1) microcanonical: one block, exact Marchenko-Pastur c1 (step3g c1_micro)
    sys.path.insert(0, '03_phase3_quantum/code')
    from step3g_decoding import c1_micro
    rows = []
    for lN, lk in ((10.0, 7.0), (10.0, 9.5), (10.0, 10.5), (10.0, 12.0), (10.0, 15.0)):
        R = ExactResolvent(np.array([lN]), [np.array([0.0])], lk)
        lJ = R.logJ()[0, 0]; c1 = np.exp(lJ*2 + 0.0 - lN + 0.0)     # rho w^2 J^2 = e^{-S} J^2
        ref = c1_micro(np.exp(lk), np.exp(lN))
        rows.append(dict(logN=lN, logk=lk, c1_exact=float(c1), c1_MP=float(ref), sum_rules=list(map(float, R.sum_rules()))))
        print(f"micro  log N={lN} log k={lk}: c1 contour {c1:.10f}  Marchenko-Pastur {ref:.10f}  sum rules {R.sum_rules()}")
    out['microcanonical'] = rows
    # (2) PSSY figs 7 and 10: beta = 3, large mu, d_code = 1
    s, lrho, lp, Sbh = pssy_blocks(3.0)
    print(f"PSSY beta=3: S_BH - S0 = {Sbh:.4f}")
    rows = []
    for x in (-6, -5, -4, -3, -2.5, -2, -1, 0, 1, 2, 2.5, 3, 4, 5, 6, 7.5):
        R = ExactResolvent(lrho, [lp], Sbh + x)
        lJ = R.logJ()[0]; lJc = R.logJ_canonical()[0]
        lw = lp - lrho
        c1 = np.exp(lse(lrho + 2*lw + 2*lJ)); c1c = np.exp(lse(lrho + 2*lw + 2*lJc))
        # entropy: exact contour and PSSY (2.52)
        Sx = R.entropy()
        lam = np.exp(np.logaddexp(R.lLam0 - R.logk, lw)); i = R.ik
        S252 = -(np.exp(lrho[:i])*lam[:i]*np.log(lam[:i])).sum()
        sr = R.sum_rules()
        rows.append(dict(x=x, c1_exact=float(c1), c1_canonical=float(c1c), S_exact=float(Sx), S_252=float(S252), sum_rules=list(map(float, sr))))
        print(f"x={x:+5.1f}: c1 exact {c1:.4f} canonical(3.45) {c1c:.4f} | S-S0 exact {Sx:.4f}  (2.52) {S252:.4f} | sum rules {sr[0]-1:+.1e} {sr[1]-1:+.1e}")
    out['pssy_beta3'] = dict(SBH_minus_S0=float(Sbh), rows=rows)
    json.dump(out, open('03_phase3_quantum/results/step3i_resolvent_validation.json', 'w'), indent=1)
