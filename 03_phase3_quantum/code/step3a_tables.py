"""Step 3a tables: symmetrised (Keldysh) one-loop variance of the clock's proper time, beta = 2 pi units
(kappa = 1/2, S - S0 = 2 pi C, y in units beta/2pi, tau in AdS units). Creation at y_d = 0, depth tau_s.
Landmarks on C1 by boundary label of the signal: horizon y_H = ln cos(tau_s); phi = 0 at y = 0; inner horizon ln sec(tau_s)."""
import numpy as np, sys, json
sys.path.insert(0, '03_phase3_quantum/code')
from step3a_oneloop import oneloop, dtau_cl, K_bdy
out = {'intervals': [], 'rate': []}
print("A. Interior intervals (C = 1; multiply var and shift by 2 pi/(S-S0))")
print(" tau_s | interval                | Dtau   | var_K   | thermal K^2/2pi | commutator | mean shift | sigma/Dtau at S-S0=100")
for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
    yH, yl = np.log(np.cos(ts)), np.log(1/np.cos(ts))
    for name, y1, y2 in [('exterior: y_H-3 -> horizon', yH - 3, yH), ('horizon -> phi=0', yH, 0.0),
                         ('horizon -> 0.9 inner', yH, 0.9*yl), ('phi=0 -> 0.9 inner', 0.0, 0.9*yl)]:
        V1, d2 = oneloop(ts, y1, y2); var = -V1; dt = dtau_cl(ts, y1, y2); K = K_bdy(ts, y1, y2)
        sh = (d2/1j).real*np.sign(1.0)                       # sign fixed by convention check below
        row = dict(ts=ts, interval=name, y1=y1, y2=y2, dtau=dt, varK=var.real, thermal=K**2/(2*np.pi), comm=var.imag, d2=(d2/1j).real)
        out['intervals'].append(row)
        print(f" {ts:.3f} | {name:24s}| {dt:.4f} | {var.real:.5f} | {K**2/(2*np.pi):.5f}         | {var.imag:+.5f}   | {(d2/1j).real:+.5f}   | {np.sqrt(var.real*2*np.pi/100)/dt:.3f}")
print("\nB. Local rate fluctuation r = lim var_K/Dtau^2 (C = 1) along the worldline, and thermal part")
for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
    yH, yl = np.log(np.cos(ts)), np.log(1/np.cos(ts))
    for name, y0 in [('y_H - 2', yH - 2), ('y_H - 1', yH - 1), ('horizon', yH), ('mid interior', yH/2), ('phi = 0', 0.0), ('0.5 inner', 0.5*yl), ('0.9 inner', 0.9*yl)]:
        d = 1e-3; V1, _ = oneloop(ts, y0 - d/2, y0 + d/2); dt = dtau_cl(ts, y0 - d/2, y0 + d/2); K = K_bdy(ts, y0 - d/2, y0 + d/2)
        r = (-V1).real/dt**2; rt = K**2/(2*np.pi*dt**2)
        out['rate'].append(dict(ts=ts, where=name, y0=y0, r=r, r_thermal=rt))
        print(f" {ts:.3f} | {name:13s} y0={y0:+.3f} | r = {r:.4f} | thermal {rt:.4f} | ratio {r/rt if rt>1e-12 else float('inf'):.1f}")
json.dump(out, open('03_phase3_quantum/results/step3a_tables.json', 'w'), indent=1, default=float)
