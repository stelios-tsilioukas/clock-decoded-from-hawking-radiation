"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.

Step 3h: decoded register density matrix in the logical basis {Psi_0, Psi_1} and its positivity.
Physical state  Psi = Psi_0 + lam Pi_1 V(y - i eps/2) Psi_0.  To leading orders
   rho_00 = <(|Psi_0><Psi_0|)_R>,  rho_10 = lam* <Psi_0|V^dag Pi_1 (|Psi_1><Psi_0|)_R|Psi_0>,  rho_11 = |lam|^2 <Psi_0|V^dag Pi_1 (|Psi_1><Psi_1|)_R Pi_1 V|Psi_0>.
Replica P-factors at shared energy s (central region):
   rho_00: y_0(s)                                 (central = arc [u_+, u_-])
   rho_10: T(s)  (one vertex on the bra line)      (central = arc [u_+, y_bra])
   rho_11: Y11(s) = mu e^{-t_B s^2} S1(s) S2(s)/F1(s),  S_i(s) = Int_D I_i(D,s),  F1(s) = Int_D g_{D'}(D,s)   (central = arc between the two signals)
with I_i the Step 3f triangles.  Decoded:  rho_ab = Sum_s rho-bar w_P w_D (lam0 + w_a)^{-1/2} (lam0 + w_b)^{-1/2}.
Positivity requires |rho_10|^2 <= rho_00 rho_11.
"""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3f_ramsey as F, step3d_backreaction as B
from step3h_replica import lse
from scipy.special import loggamma

def pieces(C, ts, y, eps, D, om, h, kmax, N):
    X = F.ExactRamsey(C, kmax, N); s = X.s; K = X.k; lw = np.log(X.w); lmu = X.lmuh + 2*np.pi*K
    tD = X.bM - 2*s*ts
    ly = {a: lmu + lw - K**2*(2*s*ts) + X.log_inner(Da, tD).real for a, Da in ((0, D), (1, D + om))}
    # rho_10: single vertex on the bra line, target <O_D(u-) V(eps/2 + i y) O_{D+om}(u+)>
    wa, wb, wc = (h + om)/2, (h - om)/2, D + (om - h)/2
    tA = s*(ts + eps/2 + 1j*y); tC = s*(ts - eps/2 - 1j*y)
    v = lmu + lw + X.log_inner(wc, tD) + X.log_inner(wb, tC)
    lT = np.array([lse(F.lg(wa, K[i], K) + v) for i in range(K.size)]) + lmu + lw - K**2*tA
    # rho_11: two vertices (ket at y - i eps/2, bra at y + i eps/2), psi_1 threads
    a, b, c = (h - om)/2, (h + om)/2, D + (om - h)/2
    t1 = s*(ts - eps/2 + 1j*y); t3 = s*(ts - eps/2 - 1j*y); tB = s*eps
    Gc, Gb = X.lghat(c), X.lghat(b); MX = X.MX
    P1 = X.logLmR(Gc, Gb, X.log_inner(a, t1)); P2 = X.logLmR(Gc, Gb, X.log_inner(a, t3))   # = log I + pi(D+B)
    lgp = F.lg(D + om, K[:, None], K[None, :])
    colD = (lmu + lw - K**2*tD)[:, None] - np.pi*K[:, None] - np.pi*K[None, :]
    S1 = lse(P1 + colD, axis=0); S2 = lse(P2 + colD, axis=0); Fq = lse(lgp + (lmu + lw - K**2*tD)[:, None], axis=0)
    lY11 = lmu + lw - K**2*tB + S1 + S2 - Fq
    # physical Ramsey norm (correlated D), for comparison
    E = P1 + P2 - 2*np.pi*(K[:, None] + K[None, :]) - lgp + (lmu + lw - K**2*tD)[:, None] + (lmu + lw - K**2*tB)[None, :]
    lF11 = lse(E[1:, 1:])
    keep = K > 0                                   # mu(0) = 0: drop the k = 0 node (zero weight, avoids log 0 - log 0)
    ly = {a: v[keep] for a, v in ly.items()}; lT, lY11, lmu, lw, K = lT[keep], lY11[keep], lmu[keep], lw[keep], K[keep]
    return X, K, lmu, lw, ly, lT, lY11, lF11

def decoded(lrho, ly, lT, lY11, logk):
    lZ0, lZ1 = lse(ly[0]), lse(ly[1]); lP0, lP1 = ly[0] - lZ0, ly[1] - lZ1
    lw0, lw1 = lP0 - lrho, lP1 - lrho
    if np.isfinite(logk):
        cum = np.logaddexp.accumulate(lrho); i = min(int(np.searchsorted(cum, logk)), lrho.size - 1)
        lam0 = np.exp(lse((np.logaddexp(lP0, lP1) - np.log(2))[i:]) - logk) if i < lrho.size - 1 else 0.0
    else:
        lam0 = 0.0
    ll0 = np.log(lam0) if lam0 > 0 else -np.inf
    a0, a1 = np.logaddexp(ll0, lw0), np.logaddexp(ll0, lw1)
    r00 = np.exp(lse(lP0 + lw0 - a0))
    r10 = np.exp(lse(lT - lZ0 + lP0 + 0.5*(lZ0 - lZ1) - lrho - 0.5*(a0 + a1)))
    r11 = np.exp(lse(lY11 - lZ0 + lP1 - lrho - a1))
    return r00, r10, r11

if __name__ == "__main__":
    ts = np.pi/4; S0 = 8.0; dcode = 2; eps = 0.05; out = {}
    for C in (10.0, 2.0):
        x1 = B.saddle(2.0/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
        kmax = C*x1 + 10*np.sqrt(C*x1) + 3*np.sqrt(2*C/eps) + 10; N = int(min(2600, max(1400, kmax/0.04)))
        for om in (0.0, 0.4):
            X, K, lmu, lw, ly, lT, lY11, lF11 = pieces(C, ts, yH, eps, 2.0, om, 1.0, kmax, N)
            lrho = np.log(dcode) + S0 + lmu + lw
            lZ0, lZ1 = lse(ly[0]), lse(ly[1])
            phys10 = np.exp(lse(lT) - 0.5*(lZ0 + lZ1)); phys11_pop = np.exp(lF11 - lZ0)
            lPb = ly[0] - lZ0; Sbh = -(np.exp(lPb)*(lPb - lrho)).sum()
            print(f"C={C} omega={om} (N={N}, kmax={kmax:.0f}): physical |rho10|^2/(rho00 rho11) with rho11 = psi_1 population F11/Z0: {abs(phys10)**2/np.real(phys11_pop):.5f}")
            rows = []
            for x in (-12, -8, -4, -2, 0, 2, 4, 8, 12, 20, np.inf):
                r00, r10, r11 = decoded(lrho, ly, lT, lY11, Sbh + x)
                margin = r00*r11 - abs(r10)**2
                rows.append(dict(x=x if np.isfinite(x) else 'inf', rho00=float(np.real(r00)), rho10=[float(r10.real), float(r10.imag)], rho11=float(np.real(r11)),
                                 det_rel=float(np.real(margin/(r00*r11)))))
                print(f"  x={x:+6.1f}: rho00={r00:.4f}  |rho10|={abs(r10):.5f} arg={np.angle(r10):+.5f}  rho11={r11:.6f}   1 - |rho10|^2/(rho00 rho11) = {margin/(r00*r11):+.5f}", flush=True)
            out[f"C{C}_om{om}"] = dict(phys_rho10=[float(phys10.real), float(phys10.imag)], phys_F11_over_Z0=float(np.real(phys11_pop)), phys_CS_ratio=float(abs(phys10)**2/np.real(phys11_pop)), rows=rows)
    json.dump(out, open('03_phase3_quantum/results/step3h_density.json', 'w'), indent=1)
