"""
Step 2D -- infalling clock C1, station clock C2 and the twin loop in the evaporating JT background.

Geometry is exactly AdS2 (only phi and f(y) evolve), so we work in rescaled, translated Poincare
coordinates centred on the drop event S:
    ah = (a - t_inf)/G(Y_S) = -g_a/G_S,   bh = (b - t_inf)/G(Y_S) = -g_b/G_S      (ds^2 = -4 dah dbh/(bh-ah)^2)
Event horizon: bh = 0.  A point (Y, r) [a = f(Y), r = g_b/g_a] has ah = -G(Y)/G_S, bh = r*ah.
Timelike geodesics of AdS2:  (ah - t0)(bh - t0) = -R^2;  ah = t0 - R e^{-lam}, bh = t0 + R e^{lam},
proper time tau = gd(lam) = 2 arctan(tanh(lam/2))  (total pi).
Station: fixed phi = phi_s. phi is linear-fractional in r at fixed Y (see step2_logjt), so
    r_s(Y) = (phi_s - n0 - C1)/(phi_s + C1 - m2)            (closed form)
Metric in (Y, r):  ds^2 = -4 w dY (r w dY - dr)/(1-r)^2,  w = u - v.
"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
from scipy.optimize import brentq
from scipy.integrate import quad

gd = lambda lam: 2*np.arctan(np.tanh(lam/2))
igd = lambda tau: np.arctanh(np.sin(tau))

class StaticJT:
    """Exact static black hole after the pulse (k = 0, kappa0 = 1): f = tanh y. Same interface as LogJT."""
    k, k0, t_inf = 0.0, 1.0, 1.0
    def kappa(self, y): return 1.0
    def lG(self, Y): return np.log(2.0) - 2*Y - np.log1p(np.exp(-2*Y))
    def pieces(self, Y):
        u = np.tanh(Y); w = 1 + u; lG = self.lG(Y)
        return dict(u=u, v=-1.0, w=w, lpsi=np.log(np.cosh(Y)), lG=lG, n0=0.0, m2=2*np.exp(lG), C1=2.0)
    def phi_r(self, Y, r, P=None):
        P = P or self.pieces(Y); return (P['n0'] + (1 + r)*P['C1'] - r*P['m2'])/(1 - r)

class Drop:
    def __init__(self, M, YS, phi_s, phi0=None):
        self.M, self.phi_s, self.YS = M, phi_s, YS
        self.lGS = M.lG(YS)
        rS, drS, wS = self.r_s(YS), self.dr_s(YS), M.pieces(YS)['w']
        self.rS = rS
        aS, bS = -1.0, -rS                                   # ah_S = -1, bh_S = r_S * ah_S
        ratio = rS - drS/wS                                  # dbh/dah along the station
        assert ratio > 0, "station not timelike"
        self.lamS = 0.5*np.log(ratio)
        self.R = (bS - aS)/(2*np.cosh(self.lamS)); self.t0 = aS + self.R*np.exp(-self.lamS)
        self.tauS = gd(self.lamS)
        # radar (Killing-like) time of S: midpoint of in/out boundary labels
        self.Yout = self.Y_of_lg(np.log(rS) + self.lGS) if rS > 0 else np.nan
        self.y_radar = 0.5*(YS + self.Yout)
        self.lamH = np.log(-self.t0/self.R)                  # event-horizon crossing (bh = 0)
        self.tauH = gd(self.lamH)
        self.phi0 = phi0

    # ---- station ----
    def r_s(self, Y):
        P = self.M.pieces(Y); return (self.phi_s - P['n0'] - P['C1'])/(self.phi_s + P['C1'] - P['m2'])
    def dr_s(self, Y, h=1e-5): return (self.r_s(Y + h) - self.r_s(Y - h))/(2*h)
    def station_dtau(self, Y):
        w = self.M.pieces(Y)['w']; r = self.r_s(Y)
        return 2*np.sqrt(w*(r*w - self.dr_s(Y)))/(1 - r)
    def station_tau(self, YE): return quad(self.station_dtau, self.YS, YE, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    def YE_of_T2(self, T2):
        return brentq(lambda Y: self.station_tau(Y) - T2, self.YS, self.YS + 50/self.M.kappa(self.YS), xtol=1e-13)

    # ---- coordinates ----
    def Y_of_lg(self, lg):                                   # boundary label with log G(Y) = lg (no underflow)
        return brentq(lambda Y: self.M.lG(Y) - lg, 1e-6, self.YS + 400/self.M.kappa(self.YS), xtol=1e-14)
    def Y_of_ah(self, ah): return self.Y_of_lg(np.log(-ah) + self.lGS)
    def point(self, lam): return self.t0 - self.R*np.exp(-lam), self.t0 + self.R*np.exp(lam)
    def phi_at(self, lam):
        ah, bh = self.point(lam); Y = self.Y_of_ah(ah); return self.M.phi_r(Y, bh/ah)
    def check_phi_S(self): return self.phi_at(self.lamS) - self.phi_s

    # ---- end of the JT interior on C1 ----
    def lam_end(self, target=0.0):
        """first lam > lamH where phi = target (phi decreasing inside)."""
        f = lambda l: self.phi_at(l) - target
        lo = self.lamH; hi = lo + 0.05
        while f(hi) > 0:
            lo, hi = hi, hi + 0.1
            if hi > self.lamH + 40: return np.nan
        return brentq(f, lo, hi, xtol=1e-13)

    # ---- twin loop ----
    def loop(self, T2):
        YE = self.YE_of_T2(T2); aE = -np.exp(self.M.lG(YE) - self.lGS)
        if aE >= self.t0: return np.nan, YE
        lamM = -np.log((self.t0 - aE)/self.R)
        return (gd(lamM) - self.tauS) - T2, YE
    def lam_of_ah(self, ah): return -np.log((self.t0 - ah)/self.R)

def static_loop(T2, ts):
    Dt = T2*np.tan(ts)
    th = 2*np.arctan((np.exp(Dt)/(1 + np.sin(ts)) - 1)/np.tan(ts))
    return np.pi/2 - ts + th - T2

if __name__ == "__main__":
    out = {}
    # ================= validation on the exact static black hole =================
    S = StaticJT(); val = {}
    for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
        D = Drop(S, YS=3.0, phi_s=2/np.sin(ts))
        le = D.lam_end(0.0)
        T2s = np.linspace(0.05, 0.9, 6)*np.log((1 + np.sin(ts))/np.cos(ts))/np.tan(ts)
        lerr = max(abs(D.loop(T)[0] - static_loop(T, ts)) for T in T2s)
        val[f'{ts:.4f}'] = dict(
            phi_S_err=float(D.check_phi_S()),
            drop_to_phi0_minus_pi2=float(gd(le) - D.tauS - np.pi/2),
            horizon_to_phi0_minus_ts=float(gd(le) - D.tauH - ts),
            Yend_minus_radar=float(D.Y_of_ah(D.point(le)[0]) - D.y_radar),
            loop_vs_closed_form=float(lerr),
            station_lapse_check=float(max(abs(D.station_dtau(Y)/2 - 1/np.tan(ts)) for Y in (0.5, 3.0, 6.0))) )
    out['static_validation'] = val
    print(json.dumps(val, indent=1))
    json.dump(out, open('02_phase2_jt/results/step2d_static_validation.json', 'w'), indent=1)
