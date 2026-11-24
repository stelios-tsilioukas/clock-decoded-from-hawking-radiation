"""
Step 3f -- two-signal (Ramsey) protocol for the matter-field clock of Step 3e.

Clock: psi_0 (Delta), psi_1 (Delta' = Delta + omega); signal V (h); cubic vertex lambda psi_1 psi_0 V.
Ramsey fringe = overlap of the two ways of exciting the clock:
    F(y1, y2) = < phi_2 | phi_1 >,   |phi_i> = Pi_1 V(y_i - i eps/2) |Psi>,   |Psi> = O_Delta(u_+)|.>,
with Pi_1 the projector on the psi_1 sector.  Positive inner product => visibility |F12|/sqrt(F11 F22) <= 1.
Boundary points (beta = 2 pi, contour order):  1 = u_+ = -tau_s,  2 = -eps/2 + i y1,  3 = +eps/2 + i y2,  4 = u_- = +tau_s.
Pi_1 keeps only the single-trace s-channel family (12) -> Delta' <- (34).

Classical (fixed geometry): 1d conformal block (Dolan-Osborn normalisation, unit OPE coefficients)
    G = x12^{-(D+h)} x34^{-(h+D)} (x24/x14)^{D12} (x14/x13)^{D34} z^{D'} 2F1(D' - D12, D' + D34; 2D'; z),
    x_ij = 2 sin((u_j - u_i)/2),  z = x12 x34/(x13 x24),  D12 = Delta - h, D34 = h - Delta.
    Heavy clock: 2F1 -> 1 and z^{D'} carries e^{-i omega Dtau}.  Finite Delta: 2F1(omega+h, omega+h; 2Delta+2omega; z).

Exact disk level (glued triangles).  The psi_1 sector of the disk Hilbert space is spanned by two-energy states
|k_B, k_D> (Lin-Maldacena-Zhao); a boundary insertion of O_{Delta'} between regions B, D has wavefunction
sqrt(Gamma_{Delta'}(k_B, k_D)) (JKMS vertex).  The three-point triangle of Step 3e is therefore
<k_B k_D| V O_Delta> * sqrt(g_{Delta'}(k_B,k_D)) = I(k_B, k_D), and
    F = Int dmu(kB) dmu(kD) e^{-tB kB^2 - tD kD^2}  I1(kB,kD) I2(kB,kD) / g_{Delta'}(kB,kD),
    I1 = Int dmu(s) g_{c}(kD,s) g_{b}(s,kB) Int dmu(kA) e^{-tA kA^2} g_{a}(s,kA),   (triangle +, y1, X)
    I2 = same with cap C (triangle X, y2, -);   a = (h - omega)/2, b = (h + omega)/2, c = Delta + (omega - h)/2.
Regions: A = (+ -> y1), B = (y1 -> y2), C = (y2 -> -), D = (- -> +).
"""
import numpy as np, mpmath as mp, sys
from scipy.special import loggamma
sys.path.insert(0, '03_phase3_quantum/code')

# ------------------------------------------------------------------ classical
def pts(ts, y1, y2, eps):
    return [-ts + 0j, -eps/2 + 1j*y1, eps/2 + 1j*y2, ts + 0j]

def logG_cl(ts, y1, y2, eps, D, om, h):
    u = pts(ts, y1, y2, eps)
    x = lambda i, j: 2*np.sin((u[j] - u[i])/2)
    lx = lambda i, j: np.log(x(i, j))
    Dp = D + om; D12 = D - h; D34 = h - D
    lz = lx(0, 1) + lx(2, 3) - lx(0, 2) - lx(1, 3)
    pre = -(D + h)*lx(0, 1) - (h + D)*lx(2, 3) + D12*(lx(1, 3) - lx(0, 3)) + D34*(lx(0, 3) - lx(0, 2))
    F = complex(mp.hyp2f1(Dp - D12, Dp + D34, 2*Dp, mp.exp(lz)))
    return pre + Dp*lz + np.log(F), lz

def ramsey_cl(ts, y1, y2, eps, D, om, h):
    """classical visibility and phase: ln F12 - (ln F11 + ln F22)/2."""
    l12, lz = logG_cl(ts, y1, y2, eps, D, om, h)
    l11, _ = logG_cl(ts, y1, y1, eps, D, om, h); l22, _ = logG_cl(ts, y2, y2, eps, D, om, h)
    return l12 - (l11 + l22)/2, lz

# ------------------------------------------------------------------ exact
def lmu(k):
    k = np.maximum(k, 1e-300)
    return np.log(2*k) + 2*np.pi*k + np.log1p(-np.exp(-4*np.pi*k)) - np.log(2)

def lg(l, a, b):
    return (2*loggamma(l + 1j*(a + b)).real + 2*loggamma(l + 1j*(a - b)).real
            - loggamma(2*l).real - np.log(2*np.pi**2))

class ExactRamsey:
    """Glued-triangle evaluation on a uniform k grid, with bounded rescaled factors:
    g(a,b) = ghat(a,b) e^{-2 pi max(a,b)},  mu(k) = muhat(k) e^{2 pi k};
    L[D,s] = ghat_c(D,s) e^{-pi|D-s|}, R[s,B] = ghat_b(s,B) e^{-pi|s-B|},  m[s] = w muhat(s) innerA(s):
    I1[D,B] = e^{-pi(D+B)} (L m R)[D,B]."""
    def __init__(self, C, kmax, N):
        self.C = C; self.bM = np.pi/C; self.s = self.bM/(2*np.pi)
        self.k = np.linspace(0, kmax, N); w = np.full(N, self.k[1] - self.k[0]); w[0] *= .5; w[-1] *= .5; self.w = w
        self.lmuh = lmu(self.k) - 2*np.pi*self.k
        K = self.k; self.MX = np.maximum(K[:, None], K[None, :]); self.AD = np.abs(K[:, None] - K[None, :])
    def lghat(self, l):
        K = self.k; return lg(l, K[:, None], K[None, :]) + 2*np.pi*self.MX
    def log_inner(self, l, t):
        """log Int dmu(k) e^{-t k^2} g_l(s,k), for every grid s."""
        K = self.k; E = (lmu(K) - K**2*t)[None, :] + lg(l, K[:, None], K[None, :])
        o = E.real.max(axis=1); return o + np.log((np.exp(E - o[:, None])*self.w[None, :]).sum(axis=1))
    def logZ(self):
        k = np.linspace(0, 2*np.pi/self.bM*3 + 40, 20000); w = np.full(k.size, k[1]-k[0]); w[0] *= .5; w[-1] *= .5
        E = lmu(k) - k**2*self.bM; o = E.max(); return o + np.log((np.exp(E - o)*w).sum())
    def logLmR(self, Gc, Gb, linner):
        L = np.exp(Gc - np.pi*self.AD); R = np.exp(Gb - np.pi*self.AD)
        lm = np.log(self.w) + self.lmuh + linner; o = lm.real.max()
        return np.log((L*np.exp(lm - o)[None, :]) @ R) + o
    def logF(self, ts, y1, y2, eps, D, om, h):
        """log of the exact Ramsey overlap / Z, in beta = 2 pi time units (factor s^{2D+2h})."""
        s = self.s; K = self.k
        a, b, c = (h - om)/2, (h + om)/2, D + (om - h)/2
        tA = s*(ts - eps/2 + 1j*y1); tB = s*(eps + 1j*(y2 - y1)); tC = s*(ts - eps/2 - 1j*y2); tD = self.bM - 2*s*ts
        key = (round(c, 12), round(b, 12), round(D + om, 12))
        if getattr(self, '_key', None) != key:
            self._G = (self.lghat(c), self.lghat(b), self.lghat(D + om)); self._key = key
        Gc, Gb, Gp = self._G
        P1 = self.logLmR(Gc, Gb, self.log_inner(a, tA)); P2 = self.logLmR(Gc, Gb, self.log_inner(a, tC))
        lw = np.log(self.w)
        E = (P1 + P2 - Gp + 2*np.pi*self.MX
             + (lw + self.lmuh - K**2*tD)[:, None] + (lw + self.lmuh - K**2*tB)[None, :])
        o = E.real.max(); return o + np.log(np.exp(E - o).sum()) - self.logZ() + (2*D + 2*h)*np.log(s)
    def ramsey(self, ts, y1, y2, eps, D, om, h):
        l12 = self.logF(ts, y1, y2, eps, D, om, h); l11 = self.logF(ts, y1, y1, eps, D, om, h); l22 = self.logF(ts, y2, y2, eps, D, om, h)
        return l12 - (l11 + l22)/2, (l12, l11, l22)

# ------------------------------------------------------------------ one loop of the full block functional
from step3e_matter_clock import oneloop as _oneloop

def block_pairs(D, om, h):
    """pair weights w_ij (factor B_ij^{-w}) of the prefactor * z^{D'} ; points 0..3 = +, y1, y2, -."""
    return {(0, 1): (h - om)/2, (2, 3): (h - om)/2, (1, 3): (h + om)/2, (0, 2): (h + om)/2, (0, 3): D - h}

ZETA = {(0, 1): .5, (2, 3): .5, (0, 2): -.5, (1, 3): -.5}       # zeta = log z = sum c_p l_p,  l = log B

def oneloop_block(ts, y1, y2, eps, D, om, h, heavy=False, **kw):
    """Coefficient of 1/C in ln <G[f]> - ln G_cl for the full single-trace block functional."""
    u = pts(ts, y1, y2, eps); up = {i: u[i] for i in range(4)}
    w = block_pairs(D, om, h)
    if heavy:
        return _oneloop(up, [(p, q, v) for (p, q), v in w.items()], **kw)
    _, lz = logG_cl(ts, y1, y2, eps, D, om, h); z = mp.exp(lz)
    a = b = om + h; c = 2*(D + om)
    F0 = mp.hyp2f1(a, b, c, z); F1 = a*b/c*mp.hyp2f1(a+1, b+1, c+1, z); F2 = a*(a+1)*b*(b+1)/(c*(c+1))*mp.hyp2f1(a+2, b+2, c+2, z)
    P1 = complex(z*F1/F0); P2 = complex(z*F1/F0 + z**2*(F2/F0 - (F1/F0)**2))
    weff = dict(w)
    for p, cp in ZETA.items(): weff[p] = weff.get(p, 0) - P1*cp           # Phi' zeta = -sum (-Phi' c_p) l_p
    L = _oneloop(up, [(p, q, v) for (p, q), v in weff.items()], **kw)
    zp = [(p, q, -cp) for (p, q), cp in ZETA.items()]                      # zeta as "F" with weights -c
    V = _oneloop(up, [(p, q, 2*v) for p, q, v in zp], **kw) - 2*_oneloop(up, zp, **kw)
    return L + 0.5*P2*V

def lnV_oneloop(ts, y1, y2, eps, D, om, h, heavy=False):
    f = lambda a, b: oneloop_block(ts, a, b, eps, D, om, h, heavy=heavy)
    return f(y1, y2) - (f(y1, y1) + f(y2, y2))/2

MONO = ['D', 'w', 'h', 'DD', 'Dw', 'Dh', 'ww', 'wh', 'hh']
def lnV_poly(ts, y1, y2, eps):
    """heavy-clock one-loop ln V as a quadratic polynomial in (Delta, omega, h): least squares on a grid (exact for a quadratic)."""
    X, Y = [], []
    for D in (1.0, 2.0, 3.5):
        for om in (0.0, 0.3, 0.7):
            for h in (0.2, 0.6, 1.1):
                x = {'D': D, 'w': om, 'h': h}
                X.append([np.prod([x[c] for c in m]) for m in MONO]); Y.append(lnV_oneloop(ts, y1, y2, eps, D, om, h, heavy=True))
    X = np.array(X, complex); Y = np.array(Y); c, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return dict(zip(MONO, c)), np.abs(X @ c - Y).max()
