"""
Step 2E -- the clock's own energy.

2E.1  Exact JT point particle.  Any vacuum dilaton is a linear form phi = (A + B(a+b) + C ab)/(b-a) <-> Z = (A,B,C),
      with invariant <Z,W> = (A_Z C_W + C_Z A_W)/2 - B_Z B_W and phi_h^2 = -<Z,Z>.
      Geodesic (a-t0)(b-t0) = -R^2 <-> unit form P = (t0^2+R^2, -t0, 1)/R, <P,P> = 1, |grad p| = 1 on the worldline.
      Junction (8piG = g): phi_in = phi_out + g m p  (p < 0 on the far side), so Z_in = Z_out + g m P and
         E_out - E_in = m <Z_out,P> + g m^2/2 = m (eps_out + eps_in)/2,   eps := <Z,P> = Killing energy per unit mass.
2E.2  Clock in an energy superposition (local gap omega).  Decoded-clock off-diagonal element
         c = Sum_E sqrt(p(E - eps0) p(E - eps1)) exp(-i omega Dtau_loop(E)),   E = exterior (final) energy,
      p = canonical Schwarzian distribution of the pre-drop black-hole energy.
"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2d_clock import StaticJT, Drop, static_loop
from scipy.integrate import quad
ip = lambda Z, W: (Z[0]*W[2] + Z[2]*W[0])/2 - Z[1]*W[1]

def check_particle():
    out = {}
    S = StaticJT(); Z = np.array([2., 0., -2.])                       # static kappa = 1, phi_h = 2, g = 1
    for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
        D = Drop(S, 3.0, 2/np.sin(ts)); G = np.exp(D.lGS)
        T0 = 1 + G*D.t0; R = G*D.R; P = np.array([T0**2 + R**2, -T0, 1.])/R
        # unit normal: analytic gradient on the worldline at several lam
        gerr = 0; jerr = 0
        for lam in np.linspace(-1, 3, 9):
            a, b = T0 - R*np.exp(-lam), T0 + R*np.exp(lam)
            pa = (b - T0)/(R*(b - a)); pb = (a - T0)/(R*(b - a))       # derivatives of p on the worldline (p = 0 there)
            gerr = max(gerr, abs(-(b - a)**2*pa*pb - 1))
            ub = R*np.exp(lam)*np.cosh(lam)                              # db/dtau
            jerr = max(jerr, abs(pa - 2*ub/(b - a)**2))                  # junction: [d_a phi] = g * Int T_aa da
        eps = ip(Z, P)
        res = dict(PP=ip(P, P) - 1, unit_normal_err=gerr, junction_err=jerr,
                   eps_minus_dtau_dy=eps - 2/np.tan(ts))
        # energy bookkeeping for a finite mass
        for m in (0.01, 0.1, 0.5):
            Zin = Z + m*P; Eout = -ip(Z, Z)/2; Ein = -ip(Zin, Zin)/2
            eps_in = ip(Zin, P)
            res[f'm={m}: (Eout-Ein) - m(eps_out+eps_in)/2'] = (Eout - Ein) - m*(eps + eps_in)/2
            res[f'm={m}: (Eout-Ein)/(m eps) - 1'] = (Eout - Ein)/(m*eps) - 1
        out[f'ts={ts:.4f}'] = {k: float(v) for k, v in res.items()}
    return out

def coherence(Sm=200.0, ts0=np.pi/4, omega=3.0, T2=None, g=None):
    """Exact (Schwarzian density of states) decoded-clock coherence vs the Gaussian predictions.
    Units: phi_r = 1, AdS2 radius 1, 8piG = g. Mean exterior black hole has kappa = 1 => S-S0 = 4 pi/g."""
    g = 4*np.pi/Sm; Cs = 1/g; kap = 1.0; beta = np.pi/kap
    phi_s = 2*kap/np.sin(ts0)
    E_of_k = lambda k: 2*k**2/g
    Ebar = E_of_k(kap); sig = np.sqrt(4*kap**3/(np.pi*g))
    def tau_s(E):                         # station depth in the exterior black hole of energy E
        return np.arcsin(np.sqrt(2*g*E)/phi_s)
    if T2 is None: T2 = 0.5*np.log((1 + np.sin(ts0))/np.cos(ts0))/np.tan(ts0)
    dtau = lambda E: static_loop(T2, tau_s(E))
    epsK = lambda E: 2*np.sqrt(g*E/2)/np.tan(tau_s(E))   # dtau/dy at the station = 2 kappa cot ts
    # canonical pre-drop distribution (Schwarzian): rho(E) ~ sinh(2 pi sqrt(2 C E)), weight e^{-beta E}
    def logp(E):
        x = 2*np.pi*np.sqrt(2*Cs*np.maximum(E, 1e-300))
        return x + np.log1p(-np.exp(-2*x)) - beta*E
    Emax = 0.999*phi_s**2/(2*g)                          # station must stay outside the horizon
    Es = np.linspace(max(Ebar - 12*sig, 1e-9), min(Ebar + 12*sig, Emax), 40001)
    lp = logp(Es); Z0 = np.log(np.trapezoid(np.exp(lp - lp.max()), Es)) + lp.max()
    # branch energies (Killing energy of the clock's internal excitation, evaluated in the exterior geometry)
    # exact junction: exterior energy E = E_in + omega*eps_out(E) + g omega^2/2 ; include the Jacobian dE_in/dE
    d = omega*epsK(Es) + g*omega**2/2
    Jac = 1 - np.gradient(d, Es)
    lp0 = logp(Es) - Z0; lp1 = logp(Es - d) - Z0 + np.log(Jac)
    w = np.exp(0.5*(lp0 + lp1))
    c = np.trapezoid(w*np.exp(-1j*omega*dtau(Es)), Es)
    BC = np.trapezoid(w, Es)
    # Gaussian/linear predictions
    hE = 1e-4*Ebar
    A = (dtau(Ebar + hE) - dtau(Ebar - hE))/(2*hE)
    K = 2*Ebar*A                                          # dDtau/dln phi_h
    delta = omega*epsK(Ebar)
    G_rem = delta**2/(8*sig**2); G_deph = omega**2*A**2*sig**2/2; Phi_sg = omega*A*delta/2
    pred_rem = (np.pi**2/2)*omega**2/np.tan(ts0)**2/Sm
    pred_deph = omega**2*K**2/(2*Sm); pred_sg = np.pi*omega**2*K/np.tan(ts0)/Sm
    # exact phase relative to the branch-0 mean geometry
    # self-gravity phase = phase of c relative to the same energy average WITHOUT the branch shift
    c_ref = np.trapezoid(np.exp(lp0)*np.exp(-1j*omega*dtau(Es)), Es)
    phase_exact = np.angle(c_ref/c)
    return dict(S_minus_S0=Sm, omega=omega, ts=ts0, T2=T2,
                minus_log_abs_c=float(-np.log(abs(c))), Gamma_rem_plus_deph=float(G_rem + G_deph),
                minus_log_BC=float(-np.log(BC)), Gamma_rem=float(G_rem), Gamma_rem_closed=float(pred_rem),
                Gamma_deph=float(G_deph), Gamma_deph_closed=float(pred_deph),
                phase_shift_exact=float(np.angle(np.exp(1j*phase_exact))), Phi_sg=float(Phi_sg),
                Phi_sg_closed=float(pred_sg), identity_Phi2_minus_4GG=float(Phi_sg**2 - 4*G_rem*G_deph), K=float(K))

if __name__ == "__main__":
    out = {'particle': check_particle()}
    print(json.dumps(out['particle'], indent=1))
    rows = []
    for Sm in (100.0, 400.0, 1600.0):
        for omega in (1.0, 3.0, 6.0):
            r = coherence(Sm=Sm, omega=omega); rows.append(r)
            print(f"S-S0={Sm:6.0f} w={omega}: -ln|c|={r['minus_log_abs_c']:.5f} vs G_rem+G_deph={r['Gamma_rem_plus_deph']:.5f} | "
                  f"G_rem={r['Gamma_rem']:.5f} (closed {r['Gamma_rem_closed']:.5f}, -lnBC {r['minus_log_BC']:.5f}) "
                  f"G_deph={r['Gamma_deph']:.5f} (closed {r['Gamma_deph_closed']:.5f}) | phase {r['phase_shift_exact']:+.5f} vs Phi_sg {r['Phi_sg']:+.5f} (closed {r['Phi_sg_closed']:+.5f})")
    out['coherence'] = rows
    json.dump(out, open('02_phase2_jt/results/step2e_particle_coherence.json', 'w'), indent=1)
