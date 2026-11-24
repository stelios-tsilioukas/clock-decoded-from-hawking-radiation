"""
Phase 2 / Step 1 -- energy-spread dephasing of the twin-loop phase in JT.

Protocol (relational, all triggers by local clocks):
  * station at fixed dilaton phi_drop; black hole has horizon dilaton phi_h,
    sin(x) = phi_h/phi_drop  (x = tau_s, C1's proper time horizon -> phi=0)
  * C1 released from rest at the station at event S
  * C2 waits proper time T2 at the station (Rindler Dt = T2 tan x), then signals inward
  * Dtau_loop(x; T2) = tau_C1(S->M) - T2
Energy: E = phi_h^2/(2 phi_r)  =>  d ln phi_h = dE/(2E);  d x / d ln phi_h = tan x.
Sensitivity K(x;T2) := d Dtau_loop / d ln phi_h |_{T2, phi_drop} = tan(x) * dDtau/dx.
Phase spread for energy spread sigma_E:  dPhi = omega K sigma_E/(2E).
Canonical JT:  sigma_E/(2E) = 1/sqrt(S - S0).
"""
import numpy as np, sympy as sp, json
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---------------- closed forms ----------------
def tau_hit(Dt, x):
    return 2*np.arctan((np.exp(Dt)/(1+np.sin(x)) - 1)/np.tan(x))
def loop(T2, x):
    return np.pi/2 - x + tau_hit(T2*np.tan(x), x) - T2
def T2_window(x):          # T2 for M on horizon / at phi=0
    return np.log(1+np.sin(x))/np.tan(x), np.log((1+np.sin(x))/np.cos(x))/np.tan(x)

checks = {}

# ---------------- 1b: independent ODE end-to-end check ----------------
def loop_ode(T2, x):
    rs = np.log(np.tan(np.pi/4 - x/2))                 # station r*
    U0, V0 = -np.exp(rs), np.exp(rs)
    tdot = np.tan(x)                                     # dt/dtau = 1/N0 at rest
    y0 = [U0, V0, np.exp(rs)*tdot, np.exp(rs)*tdot]      # U, V, U', V'
    def rhs(tau, y):
        U, V, Up, Vp = y
        return [Up, Vp, 2*V*Up**2/(1+U*V), 2*U*Vp**2/(1+U*V)]
    # station proper time T2 -> Rindler Dt from the metric lapse directly
    N0 = 1/np.sinh(abs(rs))
    Vm = np.exp(T2/N0 + rs)
    ev = lambda tau, y: y[1] - Vm; ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, [0, 4], y0, events=ev, rtol=1e-12, atol=1e-13, dense_output=True)
    tauM = sol.t_events[0][0]
    # normalisation drift along the path
    yy = sol.sol(np.linspace(0, tauM, 200))
    nd = np.max(np.abs(-4*yy[2]*yy[3]/(1+yy[0]*yy[1])**2 + 1))
    return tauM - T2, nd
err, ndmax = 0, 0
for x in (0.3, np.pi/4, 1.2):
    a, b = T2_window(x)
    for T2 in np.linspace(0.2*a, b*0.999, 7):
        v, nd = loop_ode(T2, x)
        err = max(err, abs(v - loop(T2, x))); ndmax = max(ndmax, nd)
checks['ODE vs closed-form loop'] = err
checks['ODE normalisation drift'] = ndmax

# ---------------- 1c: sensitivity, symbolic + finite difference ----------------
T, X = sp.symbols('T2 x', positive=True)
th = 2*sp.atan((sp.exp(T*sp.tan(X))/(1+sp.sin(X)) - 1)/sp.tan(X))
L = sp.pi/2 - X + th - T
Ksym = sp.tan(X)*sp.diff(L, X)
Kf = sp.lambdify((T, X), Ksym, 'numpy')
def K_fd(T2, x, h=1e-6):
    return np.tan(x)*(loop(T2, x+h) - loop(T2, x-h))/(2*h)
e2 = 0
for x in (0.3, np.pi/4, 1.2):
    a, b = T2_window(x)
    for T2 in np.linspace(a*1.01, b*0.99, 9):
        e2 = max(e2, abs(Kf(T2, x) - K_fd(T2, x))/max(1, abs(Kf(T2, x))))
checks['K symbolic vs finite-diff (rel)'] = e2

# closed form of K, simplified, at horizon and at phi=0 meeting
Kh = sp.simplify(Ksym.subs(T, sp.log(1+sp.sin(X))/sp.tan(X)))
Ks = sp.simplify(Ksym.subs(T, sp.log((1+sp.sin(X))/sp.cos(X))/sp.tan(X)))
print("K at M on horizon :", Kh)
print("K at M on phi=0   :", Ks)

# ---------------- convention: phi=0 vs Phi_total = phi0+phi = 0 ----------------
# C1's U V(tau) beyond tau=x; Phi_total=0 at UV=(1+r)/(1-r), r=phi_h/phi0
def extra_tau(x, r):
    UVt = (1+r)/(1-r)
    f = lambda t: (np.sin(t/2)/np.sin(x-t/2))*(np.cos(x-t/2)/np.cos(t/2)) - UVt
    return brentq(f, x, 2*x - 1e-12) - x
conv = {f"x={x:.3f}": {f"phi_h/phi0={r}": extra_tau(x, r) for r in (1e-1, 1e-2, 1e-3)} for x in (0.3, np.pi/4, 1.2)}
# expected linear coefficient: d(UV)/dtau at tau=x
for x in (0.3, np.pi/4, 1.2):
    tt = sp.symbols('t')
    UVe = (sp.sin(tt/2)/sp.sin(x-tt/2))*(sp.cos(x-tt/2)/sp.cos(tt/2))
    slope = float(sp.diff(UVe, tt).subs(tt, x))
    conv[f"x={x:.3f}"]["predicted extra/r (=2/slope)"] = 2/slope

print("\nchecks:")
for k, v in checks.items(): print(f"  {k:34s} {v:.2e}")
print("\nextra proper time from phi=0 to Phi_total=0:")
for k, v in conv.items(): print(" ", k, {kk: round(vv, 6) for kk, vv in v.items()})

# ---------------- 1d: visibility ----------------
def visibility_gauss(omega, K, s):         # s = sigma_E/(2E) = sigma(ln phi_h)
    return np.exp(-0.5*(omega*K*s)**2)
def visibility_exact(omega, T2, x0, s, n=4001):
    # average e^{i omega Dtau} over Gaussian ln(phi_h) with width s (exact geometry, no linearisation)
    lp = np.linspace(-6*s, 6*s, n); w = np.exp(-0.5*(lp/s)**2); w /= w.sum()
    xs = np.arcsin(np.clip(np.sin(x0)*np.exp(lp), -1, 1))
    ph = omega*loop(T2, xs)
    return abs(np.sum(w*np.exp(1j*ph)))
vis = {}
x0 = np.pi/4; a, b = T2_window(x0); T2 = 0.5*(a+b)
K0 = float(Kf(T2, x0))
for s in (1e-3, 1e-2, 3e-2):
    for om in (10, 100, 300):
        vis[f"s={s},omega={om}"] = (visibility_gauss(om, K0, s), visibility_exact(om, T2, x0, s))
print(f"\nvisibility, x=pi/4, T2 mid-window (K={K0:.4f}): (linearised Gaussian, exact average)")
for k, v in vis.items(): print(f"  {k:18s} {v[0]:.5f}  {v[1]:.5f}")
json.dump(dict(checks=checks, conv=conv, K_mid_pi4=K0,
               vis={k: list(v) for k, v in vis.items()}), open('../results/step1_dephasing.json', 'w'), indent=1)

# ---------------- figure ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for x, c in ((np.pi/8, '#1f77b4'), (np.pi/4, '#d95f02'), (3*np.pi/8, '#7570b3')):
    a, b = T2_window(x); TT = np.linspace(a, b, 300)
    depth = tau_hit(TT*np.tan(x), x)/x
    ax[0].plot(depth, Kf(TT, x), color=c, label=fr'$\tau_s={x/np.pi:.3f}\pi$')
    ax[1].plot(depth, np.sqrt(2)/np.abs(Kf(TT, x)), color=c)
ax[0].axhline(0, color='k', lw=.6)
ax[0].set_xlabel(r'meeting depth $\tau_{hit}/\tau_s$ (0 horizon, 1 $\phi=0$)')
ax[0].set_ylabel(r'$K=\partial\Delta\tau_{loop}/\partial\ln\phi_h$  [$\ell_{AdS}$]')
ax[0].set_title('Energy sensitivity of the loop (fixed $T_2$, $\\phi_{drop}$)'); ax[0].legend(fontsize=8)
ax[1].set_yscale('log')
ax[1].set_xlabel(r'meeting depth $\tau_{hit}/\tau_s$')
ax[1].set_ylabel(r'$\omega_{max}\ell_{AdS}/\sqrt{S-S_0}$  (visibility $\geq e^{-1}$)')
ax[1].set_title('Largest clock frequency that keeps the loop coherent\n(canonical energy spread)')
plt.tight_layout(); plt.savefig('../results/fig_step1_dephasing.png', dpi=150)
