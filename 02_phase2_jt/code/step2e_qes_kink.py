"""
Step 2E.3 -- quantum extremal surfaces when the clock C1 carries mass m (dilaton kink along its geodesic)
in the evaporating background.

phi = phi_evap(a,b) + mu * p(a,b) * Theta(-p),   mu = 8 pi G m  (= m in our units),
p = ((a-t0)(b-t0) + R^2)/(R(b-a)) : unit form vanishing on C1, p > 0 on the boundary ("out") side.
Coordinates: hat-coordinates of step2d (ah = (a-t_inf)/G_S, bh = (b-t_inf)/G_S); QES unknowns (Y, bh).
S/(2 pi) = phi + 2k[ log(yB - Y) + log|f(yB) - b| + 1/2 log f'(Y) - log(b - a) ].
Candidates: smooth QES on the out side (= test-clock QES of §8), smooth QES on the in side (Gyongyosi's I1),
and QES stuck on the worldline (generalised-gradient condition).
"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
from step2d_clock import Drop, gd
from scipy.optimize import root, brentq

class KinkQES:
    def __init__(self, M, D, mu):
        self.M, self.D, self.mu = M, D, mu
    def p_and_grad(self, ah, bh):
        D = self.D; t0, R = D.t0, D.R
        N = (ah - t0)*(bh - t0) + R**2; Den = R*(bh - ah)
        p = N/Den
        pa = ((bh - t0)*Den + N*R)/Den**2; pb = ((ah - t0)*Den - N*R)/Den**2
        return p, pa, pb
    def coords(self, Y, bh):
        ah = -np.exp(self.M.lG(Y) - self.D.lGS); return ah, bh
    def grad(self, Y, bh, yB, side):
        """(d_Y S, d_bh S) using the dilaton of the given side ('out' or 'in')."""
        M = self.M; P = M.pieces(Y); ah = -np.exp(P['lG'] - self.D.lGS)
        r = bh/ah                                           # g_b/g_a
        Nga = P['n0'] + (1 + r)*P['C1'] - r*P['m2']
        Na = -P['C1'] + r*P['m2']; Nb = -P['C1'] + P['m2']
        dY = P['w']*(Na*(1 - r) + Nga)/(1 - r)**2
        dbh = (Nb*(1 - r) - Nga)/((-ah)*(1 - r)**2)          # G_S d_b phi_evap
        fB = -np.exp(M.lG(yB) - self.D.lGS)                  # bh of the boundary point at yB
        k2 = 2*M.k
        gY = dY + k2*(-1/(yB - Y) - P['u'] + P['w']/(1 - r))
        gb = dbh + k2*(1/(bh - fB) - 1/(bh - ah))
        if side == 'in':
            p, pa, pb = self.p_and_grad(ah, bh)
            gY += self.mu*(-ah*P['w'])*pa                     # d_Y = (d ah/dY) d_ah,  d ah/dY = -ah w
            gb += self.mu*pb
        return np.array([gY, gb])
    def Sgen(self, Y, bh, yB, side):
        M = self.M; P = M.pieces(Y); ah = -np.exp(P['lG'] - self.D.lGS); r = bh/ah
        phi = M.phi_r(Y, r, P)
        if side == 'in': phi += self.mu*self.p_and_grad(ah, bh)[0]
        fB = -np.exp(M.lG(yB) - self.D.lGS)
        return phi + 2*M.k*(np.log(yB - Y) + np.log(abs(bh - fB)) - P['lpsi'] - np.log(bh - ah))
    def solve(self, yB, side, guess):
        r = root(lambda x: self.grad(x[0], x[1], yB, side), guess, method='hybr', options=dict(xtol=1e-14))
        if not r.success: return None
        Y, bh = r.x; ah = -np.exp(self.M.lG(Y) - self.D.lGS); p = self.p_and_grad(ah, bh)[0]
        ok = (p < 0) if side == 'in' else (p > 0)
        return dict(Y=Y, bh=bh, ah=ah, p=p, valid=bool(ok), S=self.Sgen(Y, bh, yB, side))
    # proper time on C1 (from EH crossing) at which C1 crosses the outgoing ray b = bh_Q
    def tau_on_C1_b(self, bh):
        D = self.D; return gd(np.log((bh - D.t0)/D.R)) - D.tauH
    def tau_on_C1_a(self, ah):
        D = self.D; return gd(-np.log((D.t0 - ah)/D.R)) - D.tauH

def lam_gyongyosi(D, M, mu):
    """lambda = (T_out - T_in)/(T_out + T_in) from the local quasi-static horizon values, phi_h,in^2 = phi_h,out^2 - 2 mu eps - mu^2."""
    kap = M.kappa(D.y_radar); phi_out = 2*kap
    eps = 2*kap/np.tan(np.arcsin(2*kap/D.phi_s))          # Killing energy per unit mass (verified exactly in 2E.1, static)
    phi_in = np.sqrt(phi_out**2 - 2*mu*eps - mu**2)
    return (phi_out - phi_in)/(phi_out + phi_in), eps
