"""
LogJT with a second, ingoing energy pulse at boundary time y0 (Gyongyosi et al.'s protocol), built EXACTLY:
  Sch(f,y) = -2 V(y),  V = k0^2 e^{-ky} (y < y0),  V = (k0^2 e^{-k y0} + dV) e^{-k(y-y0)} (y > y0),
  f, f', f'' continuous at y0;  the pulse is a null shell on a = a0 = f(y0) with Poincare energy E_P,
  nuD = 8 pi G E_P = 2 dV / f'(y0)   (from Sch' = -(8piG/phi_r) T_aa f'^2 integrated across the pulse).
Dilaton: N = K0 + (g_a+g_b) K1 - g_a g_b K2 with
  K0 = k Int_Y^inf w G^2 + nuD G0^2 Theta(Y<y0),  K1 = 2k0^2 t_inf + k Int_0^Y w G + nuD G0 Theta(Y>y0),
  K2 = 2k0^2 + k Int_0^Y w + nuD Theta(Y>y0),      w = Sch/f' = -2 V psi^2,  G0 = G(y0).
(The boundary condition at Y -> inf fixes K0; checked numerically below.)
"""
import numpy as np, sys
from scipy.integrate import solve_ivp
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT

class LogJTPulse(LogJT):
    def __init__(self, k, y0, lam, kappa0=1.0, ymax=None):
        self.k, self.k0, self.y0 = k, kappa0, y0; ymax = ymax or 50.0/k; self.ymax = ymax
        Vm = kappa0**2*np.exp(-k*y0); Vp = Vm*((1 + lam)/(1 - lam))**2; self.dV = Vp - Vm
        V = lambda y: kappa0**2*np.exp(-k*y) if y < y0 else Vp*np.exp(-k*(y - y0))
        self.V = V
        o = dict(rtol=1e-12, atol=1e-14, method='DOP853', dense_output=True)
        rhs = lambda y, Y: [V(y) - Y[0]**2, Y[0], V(y) - 2*Y[0]*Y[2], np.exp(-2*Y[1])]
        f1 = solve_ivp(rhs, [0, y0], [0.0, 0.0, 0.0, 0.0], **o)
        f2 = solve_ivp(rhs, [y0, ymax], f1.y[:, -1], **o)
        self._fw = (f1, f2)
        uf = lambda y: self.fwsol(y)[0]
        uT = uf(ymax); R0T = Vp*np.exp(-k*(ymax - y0))/(k*uT**2)
        rhsb = lambda y, Y: [V(y) - Y[0]**2, 2*uf(y)*Y[1] - V(y)/(uf(y) - Y[0])**2]
        b2 = solve_ivp(rhsb, [ymax, y0], [0.0, R0T], **o)
        b1 = solve_ivp(rhsb, [y0, 0], b2.y[:, -1], **o)
        self._bw = (b1, b2)
        vf = lambda y: self.bwsol(y)[0]
        rhsi = lambda y, Y: [V(y)/(uf(y) - vf(y))]
        i1 = solve_ivp(rhsi, [0, y0], [0.0], **o); i2 = solve_ivp(rhsi, [y0, ymax], i1.y[:, -1], **o)
        self._i1 = (i1, i2)
        lpsiT, fT = self.fwsol(ymax)[1], self.fwsol(ymax)[3]
        self.t_inf = fT + np.exp(-2*lpsiT)/uT
        self.c = 2*k
        self.lG0 = self.lG(y0); self.w0 = self.pieces_raw(y0)['w']
        self.nuD_G0 = 2*self.dV/self.w0                      # nuD * G0 (O(1));  nuD = 2 dV/f'(y0), G0 = f'(y0)/w0
    # piecewise dense outputs
    def fwsol(self, y): return self._fw[0].sol(y) if y <= self.y0 else self._fw[1].sol(y)
    def bwsol(self, y): return self._bw[0].sol(y) if y <= self.y0 else self._bw[1].sol(y)
    def i1sol(self, y): return self._i1[0].sol(y) if y <= self.y0 else self._i1[1].sol(y)
    def kappa(self, y): return np.sqrt(self.V(y))
    def lG(self, Y):
        u, lpsi = self.fwsol(Y)[:2]; v = self.bwsol(Y)[0]; return -2*lpsi - np.log(u - v)
    def pieces_raw(self, Y):
        u, lpsi, R2, f = self.fwsol(Y); v, R0 = self.bwsol(Y); I1 = self.i1sol(Y)[0]
        w = u - v; lG = -2*lpsi - np.log(w)
        return dict(u=u, v=v, w=w, lpsi=lpsi, lG=lG, R0=R0, R2=R2, I1=I1, f=f)
    def pieces(self, Y):
        P = self.pieces_raw(Y); w, lG = P['w'], P['lG']
        n0 = -self.c*w*P['R0']; m2 = 2*self.k0**2*np.exp(lG) - self.c*P['R2']/w
        C1 = 2*self.k0**2*self.t_inf - self.c*P['I1']
        if Y < self.y0: n0 += self.nuD_G0*np.exp(self.lG0 - lG)          # nuD G0^2 / G(Y)
        else:
            C1 += self.nuD_G0                                              # nuD G0
            m2 += self.nuD_G0*np.exp(lG - self.lG0)                        # nuD G(Y)
        P.update(n0=n0, m2=m2, C1=C1); return P

if __name__ == "__main__":
    import json
    k = 1e-3; y0 = 0.5/k
    for lam in (0.0, 0.01, 0.03):
        M = LogJTPulse(k, y0, lam) if lam > 0 else LogJT(k)
        errs = []
        for Y in np.concatenate([np.linspace(0.3, y0 - 1e-3, 15), np.linspace(y0 + 1e-3, 2.5/k, 15)]):
            P = M.pieces(Y); fp = np.exp(-2*P['lpsi'])
            # boundary condition: N(Y, g_b = G) / G = n0 + 2 C1 - m2 should equal 2 f'/G = 2 w
            errs.append(abs((P['n0'] + 2*P['C1'] - P['m2'])/(2*P['w']) - 1))
        print(f"lam={lam}: boundary-condition max rel err = {max(errs):.2e}; t_inf={M.t_inf:.12f}")
