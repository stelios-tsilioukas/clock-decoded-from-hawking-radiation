"""Step 3c: 'endless journey' (outgoing/radar labels) versus finite interior time (ingoing labels) for the SAME clock.
Convention N = e^{S0}/(4 pi^2)  (Z = e^{S0} Z_0, as in Franken-Mertens-Torres)."""
import numpy as np, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
from step3c_nonperturbative import NP
from step3b_exact_rate import Exact, R_cl
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
ts, m = np.pi/4, 2.0
d_S, d_H = 0.881374, 0.346574            # |x - y_d| of the release label S and of the horizon label (tau_s = pi/4)
out = {}
for C in (1.0, 2.0, 4.0):
    X = NP(C, ts, m)
    xs = np.concatenate([np.linspace(0, 1.0, 11), np.linspace(1.2, 4, 15), np.geomspace(4.5, 96, 22)])
    res = X.run(xs); d0 = res[0]['disk']
    R0 = Exact(C).moments(ts, 0.0, m, [1])[1].real
    lnR = CubicSpline(xs, np.log([R0*r['disk']/d0 for r in res]))          # disk <R>(x)
    W = CubicSpline(xs, [R0*r['worm']/d0 for r in res])                     # wormhole numerator (N = 1)
    Rd = lambda x: np.exp(lnR(abs(x)))
    # tail exponent of the disk rate and convergent tail integral
    n_tail = -(lnR(96) - lnR(64))/np.log(96/64)
    tail = Rd(96)*96/(n_tail - 1)
    T_in_SH = quad(Rd, d_H, d_S)[0]; T_in_int = quad(Rd, 0, d_H)[0]
    T_out_disk = quad(Rd, d_S, 96, limit=200)[0] + tail
    c2 = (4*np.pi**2)**2                                                    # 1/N^2 = c2 e^{-2 S0}
    np_int = c2*quad(lambda x: W(x), 0, d_H)[0]                             # coefficient of e^{-2S0} in dT(horizon->phi=0)
    np_SH = c2*quad(lambda x: W(x), d_H, d_S)[0]
    ramp = c2*R0*X.ramp_slope/d0*X.sc                                       # d<R>/dx at large x, coefficient of e^{-2S0}
    plat = 4*np.pi**2*R0*X.plateau_level(1.0)/d0                            # R_infinity = plat * e^{-S0}
    rel_int = [c2*W(x)/Rd(x) for x in (0.0, d_H)]
    # where does the ramp overtake the disk rate, for S0 = 10 ?
    S0 = 10.0; xg = np.geomspace(2, 96, 400); ratio = c2*np.exp(-2*S0)*W(xg)/Rd(xg)
    x_dip = float(xg[np.argmax(ratio > 1)]) if (ratio > 1).any() else None
    out[C] = dict(S_minus_S0=2*np.pi*C, R0=R0, R0_over_cl=R0/R_cl(ts, 0), tail_exponent=float(n_tail),
                  T_in_S_to_horizon=T_in_SH, T_in_horizon_to_phi0=T_in_int, T_out_S_to_horizon_disk=T_out_disk,
                  classical_S_to_horizon=np.pi/2 - ts, classical_horizon_to_phi0=ts,
                  np_rel_correction_rate_interior_coeff_e2S0=[float(v) for v in rel_int],
                  np_dT_interior_coeff_e2S0=np_int, np_dT_S_to_horizon_coeff_e2S0=np_SH,
                  ramp_rate_slope_coeff_e2S0=ramp, plateau_rate_coeff_eS0=plat, x_dip_S0_10=x_dip)
    o = out[C]
    print(f"C={C} (S-S0={2*np.pi*C:.1f}):  <R>(y_d)/R_cl = {o['R0_over_cl']:.3f}; disk tail ~ x^-{n_tail:.2f}")
    print(f"   ingoing labels : S->horizon {T_in_SH:.4f} (cl {np.pi/2-ts:.4f}); horizon->phi=0 {T_in_int:.4f} (cl {ts:.4f});"
          f"  NP shift of horizon->phi=0: {np_int:.2e} e^(-2S0);  relative rate correction {rel_int[0]:.2e}..{rel_int[1]:.2e} e^(-2S0)")
    print(f"   outgoing labels: S->horizon disk {T_out_disk:.4f};  + ramp d<R>/dx = {ramp:.2e} e^(-2S0) x ;  plateau rate R_inf = {plat:.3e} e^(-S0)"
          f"  -> T_out(Lambda) ~ R_inf*Lambda (divergent).  ramp overtakes disk at x ~ {x_dip} for S0=10", flush=True)
json.dump(out, open('03_phase3_quantum/results/step3c_journeys.json', 'w'), indent=1, default=float)
