"""
Step 3c, item 1 -- ALL corrections of relative order e^{-2 S0} to <R^p> = s^p M_p / M_0  (thermal labels).

M_p = Sum_{a,s,b,c} e^{-t0 E_a - t2 E_b - t1 E_c} |C_as|^2 |B_sb|^2 |A_sc|^2,   |O_ij|^2 = g(i,j)/N,   density = N dmu + ...
Order 1/N^2 (N = e^{S0}/(4 pi^2)):
  * six genus-0 wormholes  rho_{0,2}(E_i,E_j) = -(E_i+E_j)/(4 pi^2 sqrt(E_i E_j)(E_i-E_j)^2)
        = -(1/(4 pi^2)) f.p. 1/(k_i-k_j)^2 on the full k-plane (even integrands), i.e.
        K[H] = (1/(8 pi^2)) Int dq |q| h~(q),   h(d) = Int dsig H(sig + d/2, sig - d/2)
  * four genus-one density corrections  e^{-S0} rho_{1,1},  rho_{1,1} dE = [1/(8 pi k^4) - pi/(12 k^2)] dk  (f.p.), i.e.
        G[F] = (1/(192 pi)) Int dq |q|^3 F~(q) + (pi/48) Int dq |q| F~(q)          (checked: gives Z_{1,1}(beta))
  * the same for the normalisation M_0 (one wormhole, two genus-one terms).
Contact terms and the sine kernel are contained in the finite parts (smooth weights, thermal times).
Result:  delta<R^p>/<R^p> = c_p(x) e^{-2 S0},   c_p = (4 pi^2)^2 [ dM_p/M_p - dM_0/M_0 ]   (N = 1 units inside).
"""
import numpy as np, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
from step3b_exact_rate import lmu, lg
from scipy.interpolate import CubicSpline

import os
L, NF = float(os.environ.get("ICP_L", 24.0)), int(os.environ.get("ICP_NF", 8192))   # FFT box for edge-dominated integrands
DK = 2*L/NF; KF = (np.arange(NF) - NF//2)*DK; Q = 2*np.pi*np.fft.fftfreq(NF, d=DK); DQ = 2*np.pi/(NF*DK)
ft = lambda F: np.fft.fft(np.fft.ifftshift(F, axes=-1), axis=-1)*DK
def Kfac(F1, F2):                                    # wormhole kernel for factorised even weights
    return (np.abs(Q)*ft(F1)*ft(F2)).sum(axis=-1)*DQ/(8*np.pi**2)
def Kh(h):                                           # wormhole kernel from h(d) sampled on KF
    return (np.abs(Q)*ft(h)).sum(axis=-1)*DQ/(8*np.pi**2)
def G1(F):                                           # genus-one density functional
    Ft = ft(F); return ((np.abs(Q)**3*Ft).sum(axis=-1)/(192*np.pi) + (np.abs(Q)*Ft).sum(axis=-1)*np.pi/48)*DQ
gfun = lambda l, a, b: np.exp(lg(l, a, b))

class AllTerms:
    def __init__(self, C, ts, m, p=1.0, Nk=12001, Ns=481):
        self.C, self.ts, self.m, self.p = C, ts, m, p
        self.bM = np.pi/C; self.sc = self.bM/(2*np.pi); self.t0 = self.bM - 2*self.sc*ts
        self.kmax = 6.5/np.sqrt(self.sc*ts) + 2.0
        self.k = np.linspace(0, self.kmax, Nk); dk = self.k[1] - self.k[0]
        self.w = np.full(Nk, dk); self.w[0] *= .5; self.w[-1] *= .5
        self.s = np.linspace(0, min(self.kmax, L/2), Ns); self.ds = self.s[1] - self.s[0]
        self.ws = np.full(Ns, self.ds); self.ws[0] *= .5; self.ws[-1] *= .5
        self.mu_s = np.exp(lmu(self.s)); self.mu_s[0] = 0.0
        # A(s) = Int dmu(a) e^{-t0 a^2} g0(a,s)
        E = lmu(self.k)[None, :] + lg(m - p/2, self.s[:, None], self.k[None, :]) - (self.k**2*self.t0)[None, :]
        self.A = (np.exp(E)*self.w[None, :]).sum(axis=1)
        self.lg_sk = lg(p/2, self.s[:, None], self.k[None, :]); self.lmu_k = lmu(self.k)
        self.g0_F = gfun(m - p/2, KF[None, :], self.s[:, None])               # g0(k, s) on the FFT grid
        self.g_F = gfun(p/2, self.s[:, None], KF[None, :])                    # g(s, k) on the FFT grid
        self.Phi_a = self.g0_F*np.exp(-self.t0*KF**2)[None, :]
    def even_spline(self, y):                         # spline of an even function of s, evaluated at any real argument
        cs = CubicSpline(self.s, y, bc_type=((1, 0.0), 'not-a-knot'))
        smax = self.s[-1]
        return lambda z: np.where(np.abs(z) <= smax, cs(np.minimum(np.abs(z), smax)), 0.0)
    def K2(self, H, sig_max=10.0, Nsig=1601):
        """K[H] for a non-factorised even H(k1,k2): h(d) = Int dsig H(sig+d/2, sig-d/2)."""
        sig = np.linspace(-sig_max, sig_max, Nsig); wsig = np.full(Nsig, sig[1] - sig[0]); wsig[0] *= .5; wsig[-1] *= .5
        dsel = np.abs(KF) <= 2*sig_max
        h = np.zeros(NF, dtype=complex)
        dd = KF[dsel]
        for j0 in range(0, len(dd), 256):             # chunks to limit memory
            d = dd[j0:j0+256]
            h[np.where(dsel)[0][j0:j0+256]] = (H(sig[None, :] + d[:, None]/2, sig[None, :] - d[:, None]/2)*wsig[None, :]).sum(axis=1)
        return Kh(h)
    def run(self, Dx):
        sc, ts, p, m = self.sc, self.ts, self.p, self.m
        t1 = sc*(ts + 1j*Dx); t2 = np.conj(t1)
        E = self.lmu_k[None, :] + self.lg_sk - (self.k**2*t1)[None, :]
        B1 = (np.exp(E)*self.w[None, :]).sum(axis=1); B2 = np.conj(B1)        # B1 <-> index c, B2 <-> index b
        Phi_c = self.g_F*np.exp(-t1*KF**2)[None, :]; Phi_b = np.conj(Phi_c)
        Phi_a = self.Phi_a; W = self.ws*self.mu_s; A = self.A
        T = {}
        M = (W*A*B1*B2).sum().real; T['disk'] = M
        T['bc'] = (W*A*Kfac(Phi_b, Phi_c)).sum().real
        T['ab+ac'] = 2*(W*B1*Kfac(Phi_a, Phi_b)).sum().real
        T['Ga'] = (W*(B1*B2).real*G1(Phi_a)).sum().real/(4*np.pi**2)
        T['Gb+Gc'] = 2*(W*A*B1*G1(Phi_b)).sum().real/(4*np.pi**2)
        fA = self.even_spline(A); fBB = self.even_spline((B1*B2).real)
        fABr = self.even_spline((A*B1).real); fABi = self.even_spline((A*B1).imag)
        T['Gs'] = G1(fA(KF)*fBB(KF)).real/(4*np.pi**2)
        T['as'] = self.K2(lambda a, s: np.exp(-self.t0*a**2)*gfun(m - p/2, a, s)*fBB(s)).real
        T['sb+sc'] = 2*self.K2(lambda s, b: (fABr(s) + 1j*fABi(s))*gfun(p/2, s, b)*np.exp(-t2*b**2)).real
        return T
    def norm(self):
        """corrections to M_0 = Int dmu(a) dmu(b) e^{-t0 a^2 - tb b^2} g_m(a,b),  tb = 2 sc ts."""
        m, tb = self.m, 2*self.sc*self.ts
        E = lmu(self.k)[None, :] + lg(m, self.s[:, None], self.k[None, :])
        Ia = (np.exp(E - (self.k**2*self.t0)[None, :])*self.w[None, :]).sum(axis=1)     # Int dmu(a) e^{-t0 a^2} g_m(a, s)
        Ib = (np.exp(E - (self.k**2*tb)[None, :])*self.w[None, :]).sum(axis=1)          # Int dmu(b) e^{-tb b^2} g_m(s, b)
        W = self.ws*self.mu_s; T = {}
        T['disk'] = (W*np.exp(-tb*self.s**2)*Ia).sum()
        gm_F = gfun(m, KF[None, :], self.s[:, None])
        T['Ga'] = (W*np.exp(-tb*self.s**2)*G1(gm_F*np.exp(-self.t0*KF**2)[None, :]).real).sum()/(4*np.pi**2)
        T['Gb'] = (W*np.exp(-self.t0*self.s**2)*G1(gm_F*np.exp(-tb*KF**2)[None, :]).real).sum()/(4*np.pi**2)
        T['ab'] = self.K2(lambda a, b: np.exp(-self.t0*a**2 - tb*b**2)*gfun(m, a, b)).real
        return T

if __name__ == "__main__":
    # ---- unit tests of the functionals on Gaussians
    b1, b2 = 0.7, 1.9
    print("K (factorised) :", Kfac(np.exp(-b1*KF**2), np.exp(-b2*KF**2)).real, " exact", np.sqrt(b1*b2)/(2*np.pi*(b1+b2)))
    X0 = AllTerms(2.0, np.pi/4, 2.0)
    print("K (2D routine) :", X0.K2(lambda a, b: np.exp(-b1*a**2 - b2*b**2)).real)
    print("G1             :", G1(np.exp(-b1*KF**2)).real, " exact", (b1**1.5 + np.pi**2*b1**0.5)/(12*np.sqrt(np.pi)))
