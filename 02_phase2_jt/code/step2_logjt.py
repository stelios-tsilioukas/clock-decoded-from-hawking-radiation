"""
Overflow-free version of EvapJT (step2_aemm.py) for small k (down to ~1e-4), using only bounded variables.
  u = psi'/psi            forward Riccati   u' = V - u^2
  lpsi = log psi          lpsi' = u
  R2 = psi^-2 Int_0^y e^{-ks} psi^2 ds      R2' = e^{-ky} - 2u R2        (forward, stable)
  v = chi'/chi            backward Riccati  v' = V - v^2,  v(ymax) = 0
  R0 = psi^2 Int_y^inf e^{-ks} chi^2 ds     R0' = 2u R0 - e^{-ky}/(u-v)^2 (backward, stable)
  I1 = Int_0^y e^{-ks}/(u-v) ds
With G = t_inf - f = 1/(psi^2 (u-v)), and the ratios r = g_b/g_a, all dilaton pieces become O(1):
  n0 = C0h/g_a = -c (u-v) R0,   m2 = g_a C2 = 2 k0^2 G - c R2/(u-v),   C1 = 2 k0^2 t_inf - c I1,   c = 2 k k0^2
  phi = [n0 + (1+r) C1 - r m2]/(1-r)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

class LogJT:
    def __init__(self, k, kappa0=1.0, ymax=None):
        self.k, self.k0 = k, kappa0; ymax = ymax or 50.0/k; self.ymax = ymax
        V = lambda y: kappa0**2*np.exp(-k*y)
        o = dict(rtol=1e-12, atol=1e-14, method='DOP853', dense_output=True)
        self.fw = solve_ivp(lambda y, Y: [V(y) - Y[0]**2, Y[0], np.exp(-k*y) - 2*Y[0]*Y[2], np.exp(-2*Y[1])],
                            [0, ymax], [0.0, 0.0, 0.0, 0.0], **o)
        uf = lambda y: self.fw.sol(y)[0]
        uT = uf(ymax)
        # far-tail start for R0: Int_ymax^inf e^{-ky} chi^2 with chi ~ const => R0 ~ psi^2 e^{-k ymax} chi^2 / k
        R0T = np.exp(-k*ymax)/(k*uT**2)                     # chi^2 psi^2 = 1/(u-v)^2, v=0
        self.bw = solve_ivp(lambda y, Y: [V(y) - Y[0]**2, 2*uf(y)*Y[1] - np.exp(-k*y)/(uf(y) - Y[0])**2],
                            [ymax, 0], [0.0, R0T], **o)
        vf = lambda y: self.bw.sol(y)[0]
        self.i1 = solve_ivp(lambda y, Y: [np.exp(-k*y)/(uf(y) - vf(y))], [0, ymax], [0.0], **o)
        lpsiT, fT = self.fw.sol(ymax)[1], self.fw.sol(ymax)[3]
        self.t_inf = fT + np.exp(-2*lpsiT)/uT
        self.c = 2*k*kappa0**2

    def kappa(self, y): return self.k0*np.exp(-self.k*y/2)
    def pieces(self, Y):
        u, lpsi, R2, f = self.fw.sol(Y); v, R0 = self.bw.sol(Y); I1 = self.i1.sol(Y)[0]
        w = u - v; lG = -2*lpsi - np.log(w)
        n0 = -self.c*w*R0; m2 = 2*self.k0**2*np.exp(lG) - self.c*R2/w; C1 = 2*self.k0**2*self.t_inf - self.c*I1
        return dict(u=u, v=v, w=w, lpsi=lpsi, lG=lG, n0=n0, m2=m2, C1=C1, f=f)
    def lG(self, Y):
        u, lpsi = self.fw.sol(Y)[:2]; v = self.bw.sol(Y)[0]; return -2*lpsi - np.log(u - v)

    def phi_r(self, Y, r, P=None):
        P = P or self.pieces(Y)
        return (P['n0'] + (1 + r)*P['C1'] - r*P['m2'])/(1 - r)
    def grad(self, Y, eta, yB):
        """(f'(Y) d_a S, G_B d_b S) for S = S_gen/2pi, with g_b = -eta G_B, r = g_b/g_a."""
        P = self.pieces(Y); q = np.exp(self.lG(yB) - P['lG'])        # G_B/g_a
        r = -eta*q; Nga = P['n0'] + (1 + r)*P['C1'] - r*P['m2']
        Na = -P['C1'] + r*P['m2']; Nb = -P['C1'] + P['m2']
        dYphi = P['w']*(Na*(1 - r) + Nga)/(1 - r)**2
        dbphi = q*(Nb*(1 - r) - Nga)/(1 - r)**2
        k2 = 2*self.k
        return np.array([dYphi + k2*(-1/(yB - Y) - P['u'] + P['w']/(1 - r)),
                         dbphi + k2*(1/(1 + eta) - q/(1 - r))])
    def Sgen(self, Y, eta, yB):
        P = self.pieces(Y); q = np.exp(self.lG(yB) - P['lG']); r = -eta*q
        return (self.phi_r(Y, r, P) + 2*self.k*(np.log(yB - Y) + self.lG(yB) + np.log(abs(1 + eta))
                                               - P['lpsi'] - P['lG'] - np.log(1 - r)))
    def r_apparent(self, Y):
        P = self.pieces(Y)
        return -P['n0']/(P['C1'] + np.sqrt(P['C1']**2 + P['m2']*P['n0']))
    def qes(self, yB):
        """Robust: for each Y solve d_eta S = 0 (bracketed), then root of d_Y S along that curve."""
        def eta_star(Y):
            g = lambda e: self.grad(Y, e, yB)[1]
            es = np.concatenate([np.linspace(-0.999, 3, 120), np.geomspace(3.1, 1e4, 60)])
            vals = np.array([g(e) for e in es])
            s = np.where(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]
            return brentq(g, es[s[0]], es[s[0]+1], xtol=1e-15) if len(s) else np.nan
        kap = self.kappa(yB); D0 = np.log(max(4*kap/self.k, 2.0))/(2*kap)
        Ys = yB - np.linspace(0.2*D0, 3*D0, 60); Ys = Ys[Ys > 0.05]
        hY = []
        for Y in Ys:
            e = eta_star(Y); hY.append(self.grad(Y, e, yB)[0] if np.isfinite(e) else np.nan)
        hY = np.array(hY); ok = np.isfinite(hY)
        idx = [i for i in range(len(Ys)-1) if ok[i] and ok[i+1] and np.sign(hY[i]) != np.sign(hY[i+1])]
        if not idx: return None
        i = idx[0]
        Yq = brentq(lambda Y: self.grad(Y, eta_star(Y), yB)[0], Ys[i+1], Ys[i], xtol=1e-13)
        return Yq, eta_star(Yq)
