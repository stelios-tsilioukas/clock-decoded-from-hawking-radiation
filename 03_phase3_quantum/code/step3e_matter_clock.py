"""
Step 3e (open item 2) -- the clock as dynamical bulk matter instead of a worldline.

Model.  Two bulk fields psi_0, psi_1 of dimensions Delta and Delta + omega (the clock's two levels) and a signal field V
of dimension h, with a cubic absorption vertex  lambda psi_1 psi_0 V.  Boundary operators O_Delta, O_{Delta+omega}, V.
Boundary protocol as in Step 3b (beta = 2 pi):  u_+ = -tau_s + i y_d (clock created, level 0),
u_y = i y (signal), u_- = +tau_s + i y_d (bra, level 1).  Contour order  +, y, -.

Absorption amplitude  A_omega(y) = < O_{Delta+omega}(u_-) V(u_y) O_Delta(u_+) >.

(1) Fixed boundary curve f (any f): the cubic Witten diagram over the hyperbolic disk is the conformal three-point
    function of the points f(u_i), i.e. a triangle of Schwarzian bilocals
        A[f] = lambda C_{Delta h Delta+omega} * B(+,y)^{-w_a} B(y,-)^{-w_b} B(+,-)^{-w_c},
        w_a = (h - omega)/2,  w_b = (h + omega)/2,  w_c = Delta + (omega - h)/2,     B_ij = (f_i - f_j)^2/(f_i' f_j').
    Classically (f = tan(u/2)):  A_cl = prod (2 sin(u_ij/2))^{-2 w_ij}, and
        arg A_cl = 2 omega theta(y),  theta = arg sin((tau_s + i(y - y_d))/2),   2 theta(y) = tau(y) - tau(y_d):
    THE PHASE OF THE ABSORPTION AMPLITUDE IS omega TIMES THE GEODESIC PROPER TIME, FOR ANY Delta (no heavy limit).
    |A_cl| = R(y)^h  (up to a y-independent factor):  the modulus is the Step-3b tick rate to the power h.
(2) Exact disk level: the Schwarzian average of the triangle (Mertens-Turiaci-Verlinde rules; four regions: three caps
    and the interior k_s) -- the Step 3b integral with unequal weights:
        A/Z = Int dmu(k0) dmu(k1) dmu(k2) dmu(ks) e^{-k0^2 t0 - k1^2 t1 - k2^2 t2} g_{w_c}(k0,ks) g_{w_a}(ks,k1) g_{w_b}(ks,k2) / Z.
    Step 3b's moments are the special case omega = 0, h = p, Delta = m:  <R^p>_m = A/(norm) -- the moments of the tick
    rate ARE exact matter three-point functions of a signal of dimension p.
    Convergence: all three weights > 0, i.e.  omega < h < 2 Delta + omega  (triangle inequality of the dimensions).
(3) One loop: ln <e^F> = F_cl + <d2 F> + <(d1 F)^2>/2 with the Maldacena-Stanford-Yang propagator, general weights,
    including the eps' terms (net weights at a point no longer vanish).  Independent check of (2).
"""
import numpy as np, sys
sys.path.insert(0, '03_phase3_quantum/code')
from step3d_backreaction import lmu, lg, Exact as _Exact

# ------------------------------------------------------------------ classical
def tau_cl(ts, dy):
    """geodesic proper time at ingoing label dy = y - y_d (closed form of the audit, kappa = 1/2)."""
    return 2*np.arctan2(np.cosh(-dy/2)*np.sin(ts/2), np.sinh(-dy/2)*np.cos(ts/2))

def seps(ts, dy):
    """Euclidean contour separations of the three pairs: (+,y), (y,-), (+,-)."""
    return ts + 1j*dy, ts - 1j*dy, 2*ts + 0j

def logA_cl(ts, dy, wa, wb, wc):
    ua, ub, uc = seps(ts, np.asarray(dy, float))
    return -2*wa*np.log(2*np.sin(ua/2)) - 2*wb*np.log(2*np.sin(ub/2)) - 2*wc*np.log(2*np.sin(uc/2))

def weights(Delta, omega, h):
    return (h - omega)/2, (h + omega)/2, Delta + (omega - h)/2

# ------------------------------------------------------------------ exact (disk)
class Exact(_Exact):
    def logZ(self):
        k = np.linspace(0, 2*np.pi/self.bM*3 + 40, 20000); w = np.full(k.size, k[1]-k[0]); w[0] *= .5; w[-1] *= .5
        E = lmu(k) - k**2*self.bM; o = E.max(); return o + np.log((np.exp(E - o)*w).sum())
    def log_two_point(self, l, u, m_grid=None, ts=None):
        """log <O_l(u) O_l(0)>/Z + 2 l log s, Euclidean separation u (beta = 2 pi), for comparison with -2l log(2 sin u/2)."""
        s = self.s; k, w = self.grid(m_grid or l, ts or u/2)
        off, val = self._inner(l, s*(2*np.pi - u), k, w)
        E = lmu(k) - k**2*s*u + off; o = E.max()
        return o + np.log((np.exp(E - o)*val*w).sum()) - self.logZ() + 2*l*np.log(s)
    def log_triangle(self, ts, dys, wa, wb, wc, m_grid=None):
        """log of the exact triangle / Z + 2 (wa+wb+wc) log s, for each label dy (complex)."""
        s = self.s; t0 = self.bM - 2*s*ts; k, w = self.grid(m_grid or (wc + (wa + wb)/2), ts)
        offC, Cv = self._inner(wc, t0, k, w); dys = np.atleast_1d(dys); out = np.empty(dys.size, complex); lZ = self.logZ()
        for j, dy in enumerate(dys):
            offa, Av = self._inner(wa, s*(ts + 1j*dy), k, w)
            offb, Bv = self._inner(wb, s*(ts - 1j*dy), k, w)
            E = lmu(k) + offC + offa + offb; o = E.max()
            out[j] = o + np.log((np.exp(E - o)*Cv*Av*Bv*w).sum()) - lZ + 2*(wa + wb + wc)*np.log(s)
        return out

# ------------------------------------------------------------------ one loop, general weights
def _G(u, A, B):
    return (-(u - np.pi)**2/2 + (u - np.pi)*np.sin(u) + A + B*np.cos(u))/(2*np.pi)
def _G1(u, A, B):
    return (-(u - np.pi) + np.sin(u) + (u - np.pi)*np.cos(u) - B*np.sin(u))/(2*np.pi)
def _G2(u, A, B):
    return (-1 + 2*np.cos(u) - (u - np.pi)*np.sin(u) - B*np.cos(u))/(2*np.pi)

def oneloop(upts, pairs, A=1 + np.pi**2/6, B=2.5):
    """Coefficient of 1/C in ln <prod B_pq^{-w}> / prod_cl.   upts: dict point -> contour argument, in contour order;
    pairs: list of (p, q, w) with p before q.  Variables per point: X = eps, Y = eps'."""
    order = list(upts.keys()); idx = {p: i for i, p in enumerate(order)}; n = len(order)
    def corr(kind, a, b):          # contour-ordered <V_a W_b>, kind in XX, YX, YY
        if a == b:
            return {'XX': _G(0.0, A, B), 'YX': 0.0, 'YY': -_G2(0.0, A, B)}[kind]
        late, early = (a, b) if idx[a] > idx[b] else (b, a); d = upts[late] - upts[early]
        if kind == 'XX': return _G(d, A, B)
        if kind == 'YY': return -_G2(d, A, B)
        return _G1(d, A, B) if a == late else -_G1(d, A, B)      # <Y_a X_b>
    # first order: d1 F = sum_a (alpha_a X_a + gamma_a Y_a)
    al = {p: 0j for p in order}; ga = {p: 0j for p in order}; d2 = 0j
    for p, q, w in pairs:
        u = upts[q] - upts[p]; ct = 1/np.tan(u/2); s2 = 1/(4*np.sin(u/2)**2)
        al[q] += -w*ct; al[p] += w*ct; ga[p] += w; ga[q] += w
        # d2 F = -w [ -(X_q - X_p)^2 s2 + (Y_p^2 + Y_q^2)/2 ]
        xx = corr('XX', q, q) + corr('XX', p, p) - 2*corr('XX', p, q)
        d2 += -w*(-xx*s2 + (corr('YY', p, p) + corr('YY', q, q))/2)
    V = 0j
    for a in order:
        for b in order:
            V += al[a]*al[b]*corr('XX', a, b) + 2*ga[a]*al[b]*corr('YX', a, b) + ga[a]*ga[b]*corr('YY', a, b)
    return d2 + V/2

def oneloop_triangle(ts, dy, wa, wb, wc, **kw):
    up = {'+': -ts + 0j, 'y': 1j*dy, '-': ts + 0j}
    return oneloop(up, [('+', 'y', wa), ('y', '-', wb), ('+', '-', wc)], **kw)

def oneloop_two_point(l, u, **kw):
    return oneloop({'a': 0j, 'b': u + 0j}, [('a', 'b', l)], **kw)

# ------------------------------------------------------------------ one-loop decomposition in (Delta, omega, h)
def oneloop_quadratic(ts, dy):
    """One-loop coefficient L(w) = l.w + w.K.w/2 for the triangle (wa, wb, wc); returns (l, K)."""
    f = lambda w: oneloop_triangle(ts, dy, *w)
    E = np.eye(3); l = np.zeros(3, complex); K = np.zeros((3, 3), complex)
    for i in range(3):
        a, b = f(E[i]), f(2*E[i]); K[i, i] = b - 2*a; l[i] = a - K[i, i]/2
    for i in range(3):
        for j in range(i + 1, 3):
            K[i, j] = K[j, i] = f(E[i] + E[j]) - f(E[i]) - f(E[j])
    return l, K

WMAP = np.array([[0, -.5, .5], [0, .5, .5], [1, .5, -.5]])        # w = WMAP @ (Delta, omega, h)

def lnZhat_coeffs(ts, dy):
    """C * ln Zhat = sum of monomials in (Delta, omega, h), Zhat = A_ex/A_cl / sqrt(N_Delta N_{Delta+omega} ratios).
    Returns dict monomial -> coefficient (complex)."""
    l, K = oneloop_quadratic(ts, dy)
    lx = WMAP.T @ l; Kx = WMAP.T @ K @ WMAP                       # in x = (Delta, omega, h)
    # normalisation: -(1/2)[L(0,0,Delta) + L(0,0,Delta+omega)],  L(0,0,c) = l_c c + K_cc c^2/2
    lc, Kc = l[2], K[2, 2]
    out = {}
    names = ['D', 'w', 'h']
    for i in range(3): out[names[i]] = lx[i]
    for i in range(3):
        for j in range(i, 3):
            out[names[i] + names[j]] = Kx[i, j]*(1 if i != j else .5)
    out['D'] -= lc; out['w'] -= lc/2
    out['DD'] -= Kc/2; out['Dw'] -= Kc/2; out['ww'] -= Kc/4
    return {k: v for k, v in out.items()}

def lnZhat_1loop(ts, dy, Delta, omega, h):
    c = lnZhat_coeffs(ts, dy); x = {'D': Delta, 'w': omega, 'h': h}
    return sum(v*np.prod([x[ch] for ch in k]) for k, v in c.items())

# ------------------------------------------------------------------ physical variables: mean clock mass Dbar = Delta + omega/2
WMAP_BAR = np.array([[0, -.5, .5], [0, .5, .5], [1, 0, -.5]])     # w = WMAP_BAR @ (Dbar, omega, h)

def lnZhat_coeffs_bar(ts, dy):
    """As lnZhat_coeffs but in x = (Dbar, omega, h).  Normalisation sqrt(N_{Dbar-w/2} N_{Dbar+w/2})."""
    l, K = oneloop_quadratic(ts, dy)
    lx = WMAP_BAR.T @ l; Kx = WMAP_BAR.T @ K @ WMAP_BAR; lc, Kc = l[2], K[2, 2]
    names = ['B', 'w', 'h']; out = {}
    for i in range(3): out[names[i]] = lx[i]
    for i in range(3):
        for j in range(i, 3):
            out[names[i] + names[j]] = Kx[i, j]*(1 if i != j else .5)
    # -(1/2)[L(Dbar - w/2) + L(Dbar + w/2)] = -[lc Dbar + Kc Dbar^2/2 + Kc w^2/8]
    out['B'] -= lc; out['BB'] -= Kc/2; out['ww'] -= Kc/8
    return out

def tau_phys(tsp, dy):
    return 2*np.arctan(np.tanh(dy/2)/np.tan(tsp/2))
