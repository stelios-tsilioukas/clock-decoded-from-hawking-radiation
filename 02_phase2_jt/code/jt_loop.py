"""
Phase 2 / v0 -- interior twin loop in the JT black hole (static approximation).

Units: AdS radius = 1 (proper time), Rindler time t in units of beta/2pi.
Metric  ds^2 = -4 dU dV/(1+UV)^2,  dilaton phi = phi_h (1-UV)/(1+UV).
Horizon UV=0, boundary UV=-1, 'singularity' (phi=0) UV=1.
Exterior: U=-exp(-(t-r*)), V=exp(t+r*), ds^2=(-dt^2+dr*^2)/sinh^2 r*.

Clock C1: geodesic dropped from rest at t=0 from the station (phi_drop = phi_h/sin(tau_s)):
   U = sin(tau/2)/sin(tau_s - tau/2),  V = cos(tau_s - tau/2)/cos(tau/2)
   tau=0 horizon crossing, tau=tau_s singularity, drop at tau = tau_s - pi/2.
Clock C2: stays at the station (lapse N0 = cot tau_s) for Rindler time Dt, then its
   state rides an ingoing null ray V = const, meeting C1 at event M.
Loop phase / omega:  Dtau_loop = [pi/2 - tau_s + tau_hit(Dt)] - N0*Dt
"""
import numpy as np, sympy as sp, json
from scipy.optimize import brentq

# ---------- 1. symbolic checks of the geodesic ----------
tau, ts = sp.symbols('tau tau_s', real=True)
U = sp.sin(tau/2)/sp.sin(ts - tau/2)
V = sp.cos(ts - tau/2)/sp.cos(tau/2)
Up, Vp = sp.diff(U, tau), sp.diff(V, tau)
norm = sp.simplify(-4*Up*Vp/(1+U*V)**2)                       # should be -1
geoU = sp.diff(U, tau, 2) - 2*V*Up**2/(1+U*V)                 # geodesic eq, should be 0
geoV = sp.diff(V, tau, 2) - 2*U*Vp**2/(1+U*V)
chk = {}
pts = [(0.3, 0.9), (-0.4, 0.7), (0.05, 0.2), (-1.0, 1.2), (0.8, 1.3)]
chk['norm+1'] = max(abs(float(norm.subs({tau: a, ts: b})) + 1) for a, b in pts)
chk['geoU'] = max(abs(float(geoU.subs({tau: a, ts: b}))) for a, b in pts)
chk['geoV'] = max(abs(float(geoV.subs({tau: a, ts: b}))) for a, b in pts)
# turning point at tau = ts - pi/2: d(UV)/dtau = 0, UV = -tan^2(pi/4 - ts/2)
UV = U*V
chk['turning dUV'] = max(abs(float(sp.diff(UV, tau).subs({tau: b - sp.pi/2, ts: b}))) for _, b in pts)
chk['turning UV'] = max(abs(float((UV + sp.tan(sp.pi/4 - ts/2)**2).subs({tau: b - sp.pi/2, ts: b}))) for _, b in pts)
# dilaton at drop: phi/phi_h = (1-UV)/(1+UV) = 1/sin(ts)
chk['phi_drop'] = max(abs(float(((1-UV)/(1+UV) - 1/sp.sin(ts)).subs({tau: b - sp.pi/2, ts: b}))) for _, b in pts)
# drop event is at t=0:  U = -V there
chk['drop at t=0'] = max(abs(float((U + V).subs({tau: b - sp.pi/2, ts: b}))) for _, b in pts)
print("symbolic/numeric checks (all should be ~0):")
for k, v in chk.items(): print(f"  {k:14s} {v:.2e}")

# ---------- 2. intercept: closed form vs root finding ----------
def Vc(t, s): return np.cos(s - t/2)/np.cos(t/2)
def Uc(t, s): return np.sin(t/2)/np.sin(s - t/2)
def tau_hit(Dt, s):            # closed form, measured from horizon crossing
    return 2*np.arctan((np.exp(Dt)/(1+np.sin(s)) - 1)/np.tan(s))
def tau_hit_num(Dt, s):
    Vm = np.exp(Dt)*np.tan(np.pi/4 - s/2)          # null ray from station at time Dt
    return brentq(lambda t: Vc(t, s) - Vm, s - np.pi/2 + 1e-9, s + 0.5)
err = 0
for s in (0.2, 0.6, 1.0, 1.4):
    Dth, Dts = np.log(1+np.sin(s)), np.log((1+np.sin(s))/np.cos(s))
    for Dt in np.linspace(0.05, Dts, 9):
        err = max(err, abs(tau_hit(Dt, s) - tau_hit_num(Dt, s)))
    assert abs(tau_hit(Dth, s)) < 1e-12 and abs(tau_hit(Dts, s) - s) < 1e-12
print(f"intercept closed form vs root finding: max err {err:.1e}")

# ---------- 3. loop proper-time difference and decoding time ----------
def loop(Dt, s):               # Dtau_loop = tau_C1(S->M) - tau_C2(S->E)
    return np.pi/2 - s + tau_hit(Dt, s) - Dt/np.tan(s)
def tau_read(t, s, t_lag):
    """Latest proper time (from horizon crossing) on C1's worldline inside D(Island) at Bob's
    Rindler time t (drop at t=0). One-sided island: QES at V_Q = exp(t - t_lag) (r*_Q = -t_lag),
    D(I) contains interior points with V < V_Q.  V(tau) is increasing, so invert V(tau) = V_Q."""
    VQ = np.exp(t - t_lag)
    x = (VQ - np.cos(s))/np.sin(s)               # tan(tau/2) from V(tau)=VQ
    tr = 2*np.arctan(x)
    return np.clip(tr, s - np.pi/2, s)          # clipped to [drop, singularity]

out = {}
for s in (np.pi/8, np.pi/4, 3*np.pi/8):
    Dth, Dts = np.log(1+np.sin(s)), np.log((1+np.sin(s))/np.cos(s))
    out[f"{s:.4f}"] = dict(Dt_h=Dth, Dt_s=Dts, loop_h=float(loop(Dth, s)), loop_s=float(loop(Dts, s)))
    print(f"tau_s={s:.4f}: N0={1/np.tan(s):.3f}  window Dt in ({Dth:.4f},{Dts:.4f})  "
          f"Dtau_loop: {loop(Dth,s):+.4f} (M on horizon) -> {loop(Dts,s):+.4f} (M at singularity)")
json.dump(out, open('../results/jt_loop_v0.json', 'w'), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for s, c in ((np.pi/8, '#1f77b4'), (np.pi/4, '#d95f02'), (3*np.pi/8, '#7570b3')):
    Dth, Dts = np.log(1+np.sin(s)), np.log((1+np.sin(s))/np.cos(s))
    d_out = np.linspace(0, Dth, 100); d_in = np.linspace(Dth, Dts, 200)
    ax[0].plot(d_out, loop(d_out, s), '--', color=c)
    ax[0].plot(d_in, loop(d_in, s), '-', color=c, label=fr'$\tau_s={s/np.pi:.3f}\pi$ ($\phi_{{drop}}/\phi_h={1/np.sin(s):.2f}$)')
    ax[0].plot([Dth], [loop(Dth, s)], 'o', color=c)
    tB = np.linspace(-1, 6, 400)
    ax[1].plot(tB, tau_read(tB + 3.0, s, 3.0)/s, color=c)
ax[0].set_xlabel(r'station delay $\Delta t$  [$\beta/2\pi$]'); ax[0].set_ylabel(r'$\Delta\tau_{loop}$  [$\ell_{AdS}$]')
ax[0].set_title('Twin-loop proper-time difference\n(dashed: meeting outside; solid: inside; dot: on horizon)'); ax[0].legend(fontsize=8)
ax[1].axhline(0, color='k', lw=.6); ax[1].axhline(1, color='k', lw=.6, ls=':')
ax[1].set_xlabel(r'Bob time after drop minus $t_{lag}$  [$\beta/2\pi$]')
ax[1].set_ylabel(r'readable interior proper time $\tau_{read}/\tau_s$')
ax[1].set_title('Clock readings in $D$(island), one-sided QES\n(0 = horizon crossing, 1 = singularity)')
plt.tight_layout(); plt.savefig('../results/fig_jt_loop_v0.png', dpi=150)
