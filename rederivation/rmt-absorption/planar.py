"""
Planar (large-L, operator-valued / block-diagonal Dyson) prediction for the own-placement decoded coherence.

Notation (see report.md, Sec. 2):
  Q = [X_0/sqrt(2N_0); X_1/sqrt(2N_1)]  (2L x L, Gaussian with block variance profile)
  G = Q C = [Psi_0; Psi_1]/sqrt(2),  tau = G^dag G,  M = Q^dag Q
  atoms of the 2L space: e = (a, alpha), a in {0 (O_0), 1 (Z_1)}; brane atoms gamma; R space: one atom of size k
  sigma2[e, gamma] = A_alpha^2 B_gamma^2 s_a(alpha,gamma) / (2 N_a L)

Dyson equations (linear pencil for (s + Q C C^dag Q^dag)^{-1}) at real s>0:
  p_gamma = sum_e n_e sigma2[e,gamma] g_e
  omega   = 1/(1 + sum_gamma L_gamma p_gamma/(1 + k omega p_gamma))
  x_gamma = k omega/(1 + k omega p_gamma)            ( = s E[C (s+tau)^{-1} C^dag]_gamma )
  g_e     = 1/(s + sum_gamma L_gamma sigma2[e,gamma] x_gamma)   ( = E[(s+GG^dag)^{-1}]_e )
k = inf:  x_gamma = 1/p_gamma (omega drops out; overall C-normalisation handled by N_a ~ k).

Planar own-placement result:
  rho10_own = (eps/2)/sqrt(N0 N1) sum_{alpha,gamma} L_alpha L_gamma u(alpha,gamma) m_alpha T_{alpha gamma}
  m_alpha   = (2/pi) int ds s^{-1/2} (1 - s g_{(0,alpha)}(s))        = E[(Psi_0 tau^{-1/2} Psi_0^dag)_{ll}]
  T_{ag}    = (1/pi) int ds s^{-1/2} g_{(1,alpha)}(s) x_gamma(s)
  overlap   <Phi|Psi_1>_odd = eps k sum L L u / sqrt(N0 N1)
"""
import numpy as np
from scipy.optimize import brentq
from model import Setup


class Planar:
    def __init__(self, reg, k, fpow=1):
        S = Setup(reg, fpow=fpow)
        self.S = S
        p = S.p
        Ls = S.Ls.astype(float); Eb = S.Eb; L = S.L
        self.Ls = Ls; self.L = L; self.k = k
        A2 = np.exp(-p['tc'] * Eb)
        B2 = np.array(p['f2'])**fpow * np.exp(-p['tb'] * Eb)
        self.A2, self.B2 = A2, B2
        kk = 1.0 if k == np.inf else k
        self.kk = kk
        prof = [S.s0, S.s1]
        # N_a (planar, with E C C^dag = k)
        self.N = [kk * np.sum(Ls[:, None] * Ls[None, :] * A2[:, None] * B2[None, :] * pr) / L for pr in prof]
        sig = []
        for a in range(2):
            sig.append(A2[:, None] * B2[None, :] * prof[a] / (2 * self.N[a] * L))
        self.sig2 = np.vstack(sig)                     # (10, 5)
        self.ne = np.concatenate([Ls, Ls])             # sizes of 2L-space atoms

    def xfun(self, pg, w):
        k = self.k
        if k == np.inf:
            return 1.0 / (1.0 + pg)          # x/k in the k->inf limit (Q rescaled by sqrt(k))
        return k * w / (1 + k * w * pg)

    def solve_grid(self, s, tol=1e-12, maxit=200000):
        """vectorised damped fixed-point iteration over an array of s values"""
        sig2, ne, Ls, k = self.sig2, self.ne, self.Ls, self.k
        ns = len(s)
        g = 1.0 / s[:, None] * np.ones((ns, 10)); w = np.ones(ns)
        lam = 0.5
        for it in range(maxit):
            pg = g @ (ne[:, None] * sig2)                                   # (ns,5)
            if k != np.inf:
                wn = 1.0 / (1 + np.sum(Ls * pg / (1 + k * w[:, None] * pg), 1))
            else:
                wn = w
            x = self.xfun(pg, w[:, None])
            gn = 1.0 / (s[:, None] + x @ (Ls[:, None] * sig2.T))
            err = max(np.max(np.abs(gn / g - 1)), np.max(np.abs(wn / w - 1)))
            g = (1 - lam) * g + lam * gn
            w = (1 - lam) * w + lam * wn
            if err < tol:
                break
        pg = g @ (ne[:, None] * sig2)
        x = self.xfun(pg, w[:, None])
        return g, x, it, err

    def integrals(self, vmin=-62, vmax=34, dv=0.04):
        vs = np.arange(vmin, vmax + 1e-9, dv)
        s = np.exp(vs)
        G, X, it, err = self.solve_grid(s)
        w = s**0.5 / np.pi * dv                          # ds s^{-1/2}/pi = s^{1/2} dv/pi
        w[0] *= 0.5; w[-1] *= 0.5
        m = 2 * np.sum(w[:, None] * (1 - s[:, None] * G[:, :5]), 0)
        m1 = 2 * np.sum(w[:, None] * (1 - s[:, None] * G[:, 5:]), 0)
        t = np.sum(w[:, None] * X / s[:, None], 0)
        Tag = np.einsum('i,ia,ig->ag', w, G[:, 5:], X)
        Tu = np.einsum('i,ia,ib,ig->abg', w * s, G[:, 5:], G[:, :5], X)      # extra factor s g_{(0,beta)}(s)
        self.vs, self.G, self.X = vs, G, X
        return dict(m=m, m1=m1, t=t, T=Tag, Tu=Tu, maxerr=err, maxit=it,
                    trG0=np.sum(self.Ls * m) / 2, trG1=np.sum(self.Ls * m1) / 2)

    def wtensor(self, y):
        """w[alpha,beta,gamma] = L_a L_b L_g / L^2 * v_ab(y) s0(b,g) c_g B_g^2 ;
        v_ab = s_V(a,b) A_a a_a e^{-tau1* E_a} b_b e^{-tau2* E_b},  tau1* = tc/4 - i y, tau2* = tc/4 + i y
        (= E over V,O pairing of Y^dag_{nl} W_{ln}, summed over the middle index, per block)"""
        S = self.S; Eb = S.Eb; L = self.L; Ls = self.Ls; tc = S.p['tc']
        Aal = np.sqrt(self.A2)
        a_ = np.exp(0.5 * Eb); b_ = np.exp(-0.6 * Eb); c_ = np.exp(0.2 * Eb)
        t1c = tc / 4 - 1j * y; t2c = tc / 4 + 1j * y
        v = S.sV * (Aal * a_ * np.exp(-t1c * Eb))[:, None] * (b_ * np.exp(-t2c * Eb))[None, :]
        return (Ls[:, None, None] * Ls[None, :, None] * Ls[None, None, :] / L**2
                * v[:, :, None] * S.s0[None, :, :] * (c_ * self.B2)[None, None, :])

    def predict(self, y, eps, I=None):
        I = self.integrals() if I is None else I
        w = self.wtensor(y)
        nrm = 1 / np.sqrt(self.N[0] * self.N[1])
        overlap_conj = eps * self.kk * np.sum(w) * nrm                         # <Phi|Psi1>_odd
        own = eps / 2 * nrm * np.sum(w * I['m'][:, None, None] * I['T'][:, None, :])
        unproj = eps / 2 * nrm * np.sum(w * I['m'][:, None, None] * I['Tu'])
        D = I['m'][:, None] * I['T'] / (2 * self.kk)                           # own decoding factor (alpha,gamma)
        return dict(own=own, unproj=unproj, other=unproj - own, overlap=np.conj(overlap_conj), D=D, I=I)


if __name__ == '__main__':
    import sys
    reg = sys.argv[1] if len(sys.argv) > 1 else 'thread'
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    for k in [48, 300, 1600, 8000, 20000, np.inf]:
        P = Planar(reg, k)
        r = P.predict(y, 0.05)
        print(reg, k, 'own', np.round(r['own'], 6), 'unproj', np.round(r['unproj'], 6), 'other', np.round(r['other'], 6),
              ' <Psi1|Phi>', np.round(r['overlap'], 6), 'maxit', r['I']['maxit'])
