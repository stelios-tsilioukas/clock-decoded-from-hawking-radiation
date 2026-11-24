"""
Audit check / Step 3 groundwork: proper time of the infalling clock as the phase of a boundary cross-ratio.

Claim A (pure AdS2 geometry). A timelike geodesic (a-t0)(b-t0) = -R^2 has complex boundary endpoints
f_pm = t0 +- iR. If ingoing light rays a = a_1, a_2 (a_i < t0) meet it at proper times tau_1, tau_2, then
      exp[i(tau_2 - tau_1)] = [(f_+ - a_2)(f_- - a_1)] / [(f_- - a_2)(f_+ - a_1)]      (a cross-ratio).
Claim B (static black hole, f = tanh(kappa y)). The geodesic released at rest from the station of depth tau_s at radar
      time y_d is the one with f_+ = f(y_d + i tau_s/(2 kappa)), i.e. the particle created by a boundary operator
      displaced by Euclidean time delta = tau_s/(2 kappa).  Then, for a signal leaving the boundary at time y,
      tau(y) = 2 arctan[ tan(tau_s/2) coth(kappa (y_d - y)) ] + const,   last signal at y_d + ln sec(tau_s)/(2 kappa).
Claim C (evaporating black hole). Same identification holds up to O(k/kappa), with f continued analytically.
"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2d_clock import StaticJT, Drop, gd
from step2_logjt import LogJT
from scipy.integrate import solve_ivp
from scipy.optimize import root

out = {}
rng = np.random.default_rng(7)

# ---------------- A + B: exact static black hole (kappa = 1) ----------------
S = StaticJT(); res = {}
for ts in (np.pi/16, np.pi/8, np.pi/4, 3*np.pi/8, 0.45*np.pi):
    D = Drop(S, YS=3.0, phi_s=2/np.sin(ts)); GS = np.exp(D.lGS); yd = D.y_radar
    fp = np.tanh(yd + 1j*ts/2)                       # f(y_d + i tau_s/2)
    fp_hat = (fp - 1.0)/GS                           # in the hat coordinates of Drop
    errB = abs(fp_hat - (D.t0 + 1j*D.R))/abs(D.t0 + 1j*D.R)
    y_last = yd + 0.5*np.log(1/np.cos(ts))
    ys = np.sort(yd - rng.uniform(-0.98*(y_last - yd), 4.0, 12))   # labels from early to just before the last one
    ah = -np.exp(np.array([S.lG(y) for y in ys]) - D.lGS)
    tau = np.array([gd(D.lam_of_ah(a)) for a in ah])               # proper time from the geodesic machinery
    fpm = (D.t0 + 1j*D.R, D.t0 - 1j*D.R)
    chi = lambda a2, a1: (fpm[0] - a2)*(fpm[1] - a1)/((fpm[1] - a2)*(fpm[0] - a1))
    errA = max(abs(np.exp(1j*(tau[j] - tau[0])) - chi(ah[j], ah[0])) for j in range(1, len(ys)))
    closed = 2*np.arctan2(np.cosh(yd - ys)*np.sin(ts/2), np.sinh(yd - ys)*np.cos(ts/2))
    errC = np.max(np.abs((tau - tau[0]) - (closed - closed[0])))
    # is the stop event behind the horizon for the later labels?  (bh > 0 on C1)
    bh = D.t0 + D.R**2/(D.t0 - ah); n_inside = int(np.sum(bh > 0))
    # total proper time between past horizon (y -> -inf) and the last signal should be pi
    tot = (2*np.arctan2(np.cosh(yd - y_last)*np.sin(ts/2), np.sinh(yd - y_last)*np.cos(ts/2)) - ts)
    res[f'ts={ts:.4f}'] = dict(endpoint_err=float(errB), crossratio_err=float(errA), closed_form_err=float(errC),
                               signals_meeting_C1_behind_horizon=n_inside, total_minus_pi=float(tot - np.pi))
out['static'] = res
print(json.dumps(res, indent=1))

# ---------------- C: evaporating black hole ----------------
def complex_endpoint(M, y0, delta):
    """(f(y0 + i delta) - t_inf)/G(y0), by integrating psi'' = V psi along y = y0 + i s."""
    P = M.pieces(y0); u0, w0 = P['u'], P['w']
    V = lambda y: M.k0**2*np.exp(-M.k*y)
    rhs = lambda s, Y: [1j*Y[1], 1j*V(y0 + 1j*s)*Y[0], 1j*w0/Y[0]**2]
    sol = solve_ivp(rhs, [0, delta], [1.0 + 0j, u0 + 0j, 0j], rtol=1e-12, atol=1e-14, method='DOP853')
    return -1.0 + sol.y[2, -1]

resC = []
for k in (0.02, 0.01, 0.005):
    M = LogJT(k)
    for ts_t in (np.pi/8, np.pi/4, 3*np.pi/8):
        ydrop = 0.5/k; kap0 = M.kappa(ydrop); phi_s = 2*kap0/np.sin(ts_t)
        YS = ydrop - 0.5
        for _ in range(40):
            D = Drop(M, YS, phi_s); YS += ydrop - D.y_radar
        D = Drop(M, YS, phi_s); yr = D.y_radar; kap = M.kappa(yr); ts = np.arcsin(2*kap/phi_s)
        target = (D.t0 + 1j*D.R)*np.exp(D.lGS - M.lG(yr))            # Drop geodesic, renormalised at y_radar
        guess = complex_endpoint(M, yr, ts/(2*kap))
        naive = abs(guess - target)/abs(target + 1.0)                # compare distances from the boundary point (-1)
        # exact (y0*, delta*) reproducing the geodesic
        def F(x):
            z = complex_endpoint(M, x[0], x[1])*np.exp(M.lG(x[0]) - M.lG(yr)) - target
            return [z.real, z.imag]
        r = root(F, [yr, ts/(2*kap)], method='hybr', options=dict(xtol=1e-13))
        resC.append(dict(k=k, ts=float(ts), kappa=float(kap), naive_rel_mismatch=float(naive),
                         naive_over_k_kappa=float(naive/(k/kap)),
                         dy0_times_kappa=float(kap*(r.x[0] - yr)), ddelta_times_2kappa=float(2*kap*r.x[1] - ts),
                         dy0_coeff=float(kap*(r.x[0] - yr)/(k/kap)), ddelta_coeff=float((2*kap*r.x[1] - ts)/(k/kap)),
                         solved=bool(r.success)))
        q = resC[-1]
        print(f"k={k} ts={ts:.3f}: mismatch of f(y_radar + i ts/2kappa) = {q['naive_rel_mismatch']:.2e} "
              f"(= {q['naive_over_k_kappa']:.3f} k/kappa);  exact insertion: kappa*dy0 = {q['dy0_coeff']:+.3f} k/kappa, "
              f"2kappa*delta - ts = {q['ddelta_coeff']:+.3f} k/kappa", flush=True)
out['evaporating'] = resC
json.dump(out, open('03_phase3_quantum/results/step3_crossratio.json', 'w'), indent=1)
