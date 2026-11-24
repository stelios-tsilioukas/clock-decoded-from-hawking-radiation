"""
*** NOTE (17 Nov 2026, Step 3i): this code sums only the THREADING pairing of the clock lines. At tau_s = pi/4 the cap
*** pairing dominates by tan(tau_s/2)^{-4 Delta} (e^7 at Delta = 2): the clock is in B's wedge and is not decodable from R
*** in the static PSSY model. Tables produced at tau_s = pi/4 are withdrawn; the formulas are the leading planar term for
*** island clocks (tau_s > pi/2). See step3i_exact_decoding.md.
Step 3h: decoded clock reading vs the physical (Step 3e) reading.
reading = [arg rho_10(omega) - arg rho_10(0)]/omega (omega-dependent part of the phase), h = 1, Delta = 2, horizon label.
(a) deep post-Page: which-energy mismatch C_deep vs C_phys, scaling with omega and C;
(b) across the Page transition."""
import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_replica as H, step3d_backreaction as B
ts = np.pi/4; D = 2.0; h = 1.0; out = {'deep': [], 'scan': {}}
print("(a) deep post-Page: C*[ln C_deep - ln C_phys]  (Re: modulus, Im: phase)")
for C in (10.0, 20.0, 40.0):
    X = H.Replica(C, ts, D); x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    ly0 = X.log_y(D)
    for om in (0.1, 0.2, 0.4):
        ly1 = X.log_y(D + om); lT = X.log_T(yH, (h + om)/2, (h - om)/2, D + (om - h)/2)
        _, Cp, Cdp, _, _ = X.decode(lT, ly0, ly1, np.inf)
        z = C*np.log(Cdp/Cp); out['deep'].append(dict(C=C, omega=om, C_ln_ratio=[z.real, z.imag]))
        print(f"  C={C:4.0f} omega={om}: {z.real:+.5f} {z.imag:+.5f}i   /omega: {z.real/om:+.4f} {z.imag/om:+.4f}i   /omega^2: {z.real/om**2:+.4f} {z.imag/om**2:+.4f}i", flush=True)
print("(b) across the Page transition (C = 10 and 2): decoded reading vs physical")
for C in (10.0, 2.0):
    X = H.Replica(C, ts, D); x1 = B.saddle(D/C, ts)[1]; yH = B.dy_horizon(x1*ts)/x1
    ly0 = X.log_y(D); om = 0.2
    ly1 = X.log_y(D + om)
    lT0 = X.log_T(yH, h/2, h/2, D - h/2); lT1 = X.log_T(yH, (h + om)/2, (h - om)/2, D + (om - h)/2)
    P = np.exp(ly0 - H.lse(ly0)); Sbh = -(P*(np.log(np.maximum(P, 1e-300)) - X.lrho)).sum()
    rows = []
    _, Cp0, _, _, _ = X.decode(lT0, ly0, ly0, np.inf); _, Cp1, _, _, _ = X.decode(lT1, ly0, ly1, np.inf)
    read_phys = (np.angle(Cp1) - np.angle(Cp0))/om
    for x in (-12, -8, -4, -2, 0, 2, 4, 8, 12, 20, np.inf):
        Cd0, _, _, c1, _ = X.decode(lT0, ly0, ly0, Sbh + x); Cd1, _, _, _, _ = X.decode(lT1, ly0, ly1, Sbh + x)
        rd = (np.angle(Cd1) - np.angle(Cd0))/om
        rows.append(dict(x=x if np.isfinite(x) else 'inf', c1=c1, reading_dec=rd, reading_phys=read_phys, gap_free_phase=float(np.angle(Cd0))))
        print(f"  C={C} x={x:+6.1f}: c1={c1:.3f}  decoded reading {rd:.4f}  physical {read_phys:.4f}  (dec/phys - 1 = {rd/read_phys - 1:+.4f});  gap-free decoded phase {np.angle(Cd0):+.4f}", flush=True)
    out['scan'][str(C)] = rows
json.dump(out, open('03_phase3_quantum/results/step3h_reading.json', 'w'), indent=1, default=float)
