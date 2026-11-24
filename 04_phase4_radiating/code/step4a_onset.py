"""
Step 4a (Option B): radiating PSSY model -- when does an INFALLING clock become decodable from early + late radiation?

Model: PSSY black hole (early radiation R1, dim k1) + clock created from outside (tau_s < pi/2) + late emission channel
  E(rho) = sum_j K_j rho K_j^dag,  K_j ~ O_j(t),  j = 1..k2 independent light flavours (dimension De), Euclidean width eps_e,
  emitted at boundary time t after the drop; environment R2 (dim k2).  Bob holds R1 R2.
Replica pairings (connected geometry), per replica: clock line caps (C) or threads (T); emission line caps (c) or threads (t).
  Emission insertions sit inside the clock's middle arc (emission after the drop), hence:
   (C,c), (T,t), (T,c) are non-crossing and t-INDEPENDENT (boost invariance: every region's Lorentzian lengths cancel);
   (C,t) crosses: each threaded emission chord crosses the capped clock chords -> OTOC, the only t-dependence.
  R2 index counting: emission cap costs 1/k2 relative to thread.
Probe-limit n = 2 weights (two-point functions on the doubled disk; ratios only):
   thread/cap for the clock: theta = tan(tau_s/2)^{4 Delta};  for the emission: theta_e = tan(eps_e/2)^{4 De}
   W(C,c) = 1/k2,  W(T,t) = theta*theta_e,  W(T,c) = theta/k2,  W(C,t) = theta_e * |F(z)|,
   F(z) = E_Q[(1 - i z Q)^{-4 De}],  Q ~ Gamma(4 Delta)   (eikonal: two emission chords x two clock chords, equal couplings),
   z = e^{2 pi (t - t_*)/beta_eff},  t_* defined by z = 1;  beta_eff = n beta (Renyi-n), -> beta for decoding (n -> 1).
Decodable (T,t dominant) iff  (i) k2 > 1/(theta theta_e)   [late-radiation threshold, in nats: log k2 > 4 Delta log cot(tau_s/2) + 4 De log cot(eps_e/2)]
                              (ii) |F(z)| < theta           [scrambling onset].
"""
import numpy as np, json
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.optimize import brentq

def F(z, D, De):
    a = 4*D; f = lambda Q, part: np.exp((a-1)*np.log(Q) - Q - gammaln(a)) * (part((1 - 1j*z*Q)**(-4*De)))
    re = quad(f, 0, np.inf, args=(np.real,), limit=400)[0]; im = quad(f, 0, np.inf, args=(np.imag,), limit=400)[0]
    return re + 1j*im

def z_dec(D, De, ts):
    th = np.tan(ts/2)**(4*D)
    g = lambda lz: np.log(abs(F(np.exp(lz), D, De))) - np.log(th)
    return np.exp(brentq(g, -5, 60))

out = []
print("check F(0)=1:", F(0.0, 2, 1))
print(f"{'Delta':>5} {'De':>4} {'tau_s/pi':>8} | {'log theta':>9} | {'z_dec':>10} | Delta t_dec = (beta/2pi) log z_dec | late-radiation threshold log k2 (eps_e = pi/8)")
for D in (1.0, 2.0, 4.0):
    for De in (0.5, 1.0, 2.0):
        for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
            zd = z_dec(D, De, ts); th = np.tan(ts/2)**(4*D); eps_e = np.pi/8
            thr = 4*D*np.log(1/np.tan(ts/2)) + 4*De*np.log(1/np.tan(eps_e/2))
            row = dict(Delta=D, De=De, tau_s=ts, log_theta=float(np.log(th)), z_dec=float(zd), dt_over_beta2pi=float(np.log(zd)),
                       logk2_threshold=float(thr))
            out.append(row)
            print(f"{D:5.1f} {De:4.1f} {ts/np.pi:8.3f} | {np.log(th):+9.3f} | {zd:10.3g} | {np.log(zd):+7.3f}  (asymptote (Delta/De) log cot(tau_s/2) = {D/De*np.log(1/np.tan(ts/2)):+.3f}) | {thr:.2f}")
json.dump(out, open('04_phase4_radiating/results/step4a_onset.json','w'), indent=1)
