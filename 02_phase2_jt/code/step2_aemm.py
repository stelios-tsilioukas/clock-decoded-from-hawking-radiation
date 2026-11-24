"""
Step 2 -- evaporating JT black hole (AEMM-type set-up), built from first principles.

Conventions (field equations verified symbolically in step2_symbolic.py):
  Poincare AdS2, a = t - z (ingoing label), b = t + z (outgoing label); ds^2 = -4 da db/(b-a)^2.
  Units: phi_r = 1, 8 pi G = 1, kappa0 = 1 by default.  k = cG/(3 phi_r) = c/(24 pi).
  Boundary curve t = f(y).  f = y for y < 0; energy pulse at y = 0 (8 pi G E_S = 2 kappa0^2);
  afterwards Sch(f,y) = -2 kappa0^2 e^{-k y}  (energy balance dE/dy = -k E).
  Instantaneous kappa(y) = kappa0 e^{-k y/2};  T(y) = kappa/pi.

Numerically stable representation (no overflow / no catastrophic cancellation):
  psi'' = V psi, V = kappa0^2 e^{-ky}, psi(0)=1, psi'(0)=0;  f' = 1/psi^2.
  u = psi'/psi   (Riccati u' = V - u^2, integrated forward, stable)
  v = chi'/chi   (decaying solution chi, Riccati integrated backward, stable)
  Wronskian psi chi' - psi' chi = -1  =>  G(y) := t_inf - f(y) = f'(y)/(u - v).
Dilaton for a = f(Y) > 0, written with g_a = t_inf - a, g_b = t_inf - b:
  phi = N/(b-a),  N = C0h(Y) + (g_a+g_b) C1(Y) - g_a g_b C2(Y)
  C0h(Y) = -2 k kappa0^2 Int_Y^inf e^{-ky} chi^2 dy,   chi = 1/(psi (u-v))
  C1(Y)  = 2 kappa0^2 t_inf - 2 k kappa0^2 Int_0^Y e^{-ky} psi chi dy
  C2(Y)  = 2 kappa0^2       - 2 k kappa0^2 Int_0^Y e^{-ky} psi^2 dy
  (This is the source solution phi = [2 - 2kappa0^2 ab - k Int_0^Y w (a-f)(b-f)]/(b-a),
   w = Sch/f', regrouped; the regrouping uses the identity 2 - 2 kappa0^2 t_inf^2 = k Int_0^inf w G^2,
   which is checked numerically below.)
"""
import numpy as np, json
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import root, brentq

class EvapJT:
    def __init__(self, k, kappa0=1.0, ymax=None, n=400001):
        self.k, self.k0 = k, kappa0
        ymax = ymax if ymax else 45.0/k
        ys = np.linspace(0, ymax, n)
        V = lambda y: kappa0**2*np.exp(-k*y)
        fw = solve_ivp(lambda y, Y: [V(y) - Y[0]**2, Y[0], np.exp(-2*Y[1])],
                       [0, ymax], [0.0, 0.0, 0.0], t_eval=ys, rtol=1e-12, atol=1e-14, method='DOP853')
        u, lpsi, f = fw.y
        vb = 0.0 if k > 0 else -kappa0                      # linear tail (k>0) / exact static (k=0)
        bw = solve_ivp(lambda y, Y: [V(y) - Y[0]**2], [ymax, 0], [vb], t_eval=ys[::-1],
                       rtol=1e-12, atol=1e-14, method='DOP853')
        v = bw.y[0][::-1]
        self.y, self.u, self.v, self.lpsi, self.f = ys, u, v, lpsi, f
        self.psi2 = np.exp(2*lpsi)
        self.lG = -2*lpsi - np.log(u - v)                    # log G
        self.t_inf = f[-1] + np.exp(self.lG[-1])
        e = np.exp(-k*ys); chi = 1/(np.exp(lpsi)*(u - v)); psichi = 1/(u - v)
        c = 2*k*kappa0**2
        # integrals via spline antiderivatives (4th-order accurate, derivative-consistent)
        A = lambda arr: CubicSpline(ys, arr).antiderivative()
        Ipc, Ip2 = A(e*psichi), A(e*self.psi2)
        # tail integral accumulated from the far end (avoids I(ymax) - I(y) cancellation)
        Itail = CubicSpline(-ys[::-1], (e*chi**2)[::-1]).antiderivative()
        self.C0h = -c*Itail(-ys)
        self.C1 = 2*kappa0**2*self.t_inf - c*Ipc(ys)
        self.C2 = 2*kappa0**2 - c*Ip2(ys)
        S = lambda arr: CubicSpline(ys, arr)
        self.sU, self.sLP, self.sF, self.sLG = S(u), S(lpsi), S(f), S(self.lG)
        self.sC0, self.sC1, self.sC2 = S(self.C0h), S(self.C1), S(self.C2)
        self.sYofLG = CubicSpline(-self.lG, ys)              # inverse: -log G increasing in y

    # elementary functions of boundary time
    def G(self, Y): return np.exp(self.sLG(Y))
    def psi2f(self, Y): return np.exp(2*self.sLP(Y))
    def kappa(self, y): return self.k0*np.exp(-self.k*y/2)
    def Y_of_a(self, a): return float(self.sYofLG(-np.log(self.t_inf - a)))

    # dilaton in (Y, g_b) variables
    def N(self, Y, gb):
        ga = self.G(Y); return self.sC0(Y) + (ga + gb)*self.sC1(Y) - ga*gb*self.sC2(Y)
    def phi(self, Y, gb): return self.N(Y, gb)/(self.G(Y) - gb)
    def dphi(self, Y, gb):
        ga = self.G(Y); d = ga - gb; N = self.N(Y, gb)
        Na = -self.sC1(Y) + gb*self.sC2(Y); Nb = -self.sC1(Y) + ga*self.sC2(Y)
        return (Na*d + N)/d**2, (Nb*d - N)/d**2               # d_a phi, d_b phi

    # generalized entropy / (2 pi):  phi + (c/6)/(2pi) * log(...),  (c/6)/(2pi) = 2k
    def Sgen(self, Y, gb, yB):
        GB = self.G(yB); d = self.G(Y) - gb
        return self.phi(Y, gb) + 2*self.k*(np.log(yB - Y) + np.log(abs(gb - GB))
                                           - self.sLP(Y) - np.log(d))
    def gradS_scaled(self, Y, eta, yB):
        """(d_Y S, G_B d_b S) with g_b = -eta G_B; d_Y = f'(Y) d_a."""
        GB = self.G(yB); gb = -eta*GB; ga = self.G(Y); d = ga - gb; p2 = self.psi2f(Y)
        pa, pb = self.dphi(Y, gb)
        ga_ = pa/p2 + 2*self.k*(-1/(yB - Y) - self.sU(Y) + 1/(p2*d))
        gb_ = GB*pb + 2*self.k*(GB/(GB - gb) - GB/d)
        return np.array([ga_, gb_])
    def qes(self, yB, guess=None):
        if guess is None:
            kap = self.kappa(yB); D = np.log(4*kap/self.k)/(2*kap); guess = (yB - D, 1.0)
        r = root(lambda x: self.gradS_scaled(x[0], x[1], yB), guess, method='hybr',
                 options=dict(xtol=1e-13))
        return r.x, r.success
    def apparent_gb(self, Y):
        """g_b of the apparent horizon (d_a phi = 0) on the ingoing ray a = f(Y)."""
        ga = self.G(Y); C1, C2, C0 = self.sC1(Y), self.sC2(Y), self.sC0(Y)
        # (Na d + N) = 0 with Na = -C1 + gb C2, d = ga - gb: quadratic in gb
        # (-C1 + gb C2)(ga - gb) + C0 + (ga+gb)C1 - ga gb C2 = -C2 gb^2 + 2 C1 gb + C0 = 0
        disc = C1**2 + C2*C0
        return (C1 - np.sqrt(disc))/C2                       # root continuously connected to gb = 0

def run_checks():
    out = {}
    # ---------- static limit ----------
    M = EvapJT(k=0.0, ymax=30, n=60001)
    yy = np.linspace(0.3, 12, 40)
    out['k=0: |f - tanh y|'] = float(np.max(np.abs(M.sF(yy) - np.tanh(yy))))
    out['k=0: |G - (1 - tanh y)|/G'] = float(np.max(np.abs(M.G(yy)/(1 - np.tanh(yy)) - 1)))
    out['k=0: t_inf - 1'] = float(M.t_inf - 1)
    out['k=0: QES (should not exist w/ k=0 -> skip)'] = None
    for k in (0.01, 0.02, 0.05):
        E = EvapJT(k=k); tag = f'k={k}'
        # 2A. Schwarzian of f by finite differences (independent of the Riccati form)
        dd = solve_ivp(lambda y, Y: [Y[1], E.k0**2*np.exp(-k*y)*Y[0], 1/Y[0]**2], [0, 6/k],
                       [1.0, 0.0, 0.0], rtol=1e-13, atol=1e-15, method='DOP853', dense_output=True)
        y0 = np.linspace(0.5, 4.0, 15); h = 1e-2; F = lambda x: dd.sol(x)[2]
        d1 = (F(y0+h)-F(y0-h))/(2*h); d2 = (F(y0+h)-2*F(y0)+F(y0-h))/h**2
        d3 = (F(y0+2*h)-2*F(y0+h)+2*F(y0-h)-F(y0-2*h))/(2*h**3)
        out[tag+': Sch(fd)/exact - 1'] = float(np.max(np.abs((d3/d1 - 1.5*(d2/d1)**2)/(-2*np.exp(-k*y0)) - 1)))
        # Schwarzian from Riccati variables at all times: Sch = -2(u' + u^2) + ... check u' = V - u^2 implied;
        # check G' = -f' (G is t_inf - f) via spline derivative, in log form: (log G)' = -f'/G = -(u - v)
        yy = np.linspace(1, 2*np.log(10)/k, 50)
        out[tag+': (logG)\' + (u-v) rel'] = float(np.max(np.abs(E.sLG(yy, 1) + (E.sU(yy) - np.interp(yy, E.y, E.v)))/(E.sU(yy) - np.interp(yy, E.y, E.v))))
        out[tag+': t_inf'] = float(E.t_inf)
        # identity 2 - 2 k0^2 t_inf^2 = C0h(0)   (needed for the regrouping)
        out[tag+': identity C0h(0) - (2 - 2 t_inf^2)'] = float(E.sC0(0.0) - (2 - 2*E.t_inf**2))
        # 2B. boundary condition N(Y, g_b = G(Y)) = 2 f'(Y), relative
        Ys = np.linspace(0.2, 2*np.log(10)/k, 60)   # kappa(Y) >= kappa0/10
        bc = [abs(E.N(Y, E.G(Y))*E.psi2f(Y)/2 - 1) for Y in Ys]
        out[tag+': boundary cond. rel err'] = float(np.max(bc))
        # 2B. bulk equations by finite differences in (Y, g_b):  d_a = psi^2 d_Y,  d_b = -d_{g_b}
        rng = np.random.default_rng(3); r_ab = []; r_aa = []; r_da = []
        for _ in range(25):
            Y = rng.uniform(0.3, 8.0); ga = E.G(Y); gb = ga*rng.uniform(-3, 0.7); hY = 1e-4; hb = 1e-4*ga
            P = lambda Y_, gb_: E.phi(Y_, gb_)
            dYdb = -(P(Y+hY, gb+hb) - P(Y+hY, gb-hb) - P(Y-hY, gb+hb) + P(Y-hY, gb-hb))/(4*hY*hb)
            d = ga - gb; p2 = E.psi2f(Y)
            r_ab.append(abs(2*p2*dYdb + 4*P(Y, gb)/d**2)/(4*abs(P(Y, gb))/d**2))
            # E_aa = -d_a^2 phi + 2/(b-a) d_a phi  should equal 8piG T_aa = k Sch(Y)/f'(Y)^2 = k Sch psi^4
            # with d_a = psi^2 d_Y at fixed b (g_b fixed): d_a^2 = psi^2 d_Y(psi^2 d_Y)
            def phia(Yq): return E.dphi(Yq, gb)[0]          # analytic d_a phi; one FD below
            paa = p2*(phia(Y+hY) - phia(Y-hY))/(2*hY)
            Eaa = -paa + 2/d*phia(Y); Taa = k*(-2*E.k0**2*np.exp(-k*Y))*p2**2
            r_aa.append(abs(Eaa/Taa - 1))
            # also check analytic d_a phi against FD of phi
            r_da.append(abs(phia(Y)/(p2*(P(Y+hY, gb) - P(Y-hY, gb))/(2*hY)) - 1))
        out[tag+': bulk E_ab rel residual'] = float(np.max(r_ab))
        out[tag+': bulk E_aa / (8piG T_aa) - 1'] = float(np.max(r_aa))
        out[tag+': analytic d_a phi vs FD'] = float(np.max(r_da))
    return out

if __name__ == "__main__":
    out = run_checks()
    print(json.dumps(out, indent=1))
    json.dump(out, open('02_phase2_jt/results/step2_checks_AB.json', 'w'), indent=1)
