"""Extrapolations of the exact-disk coefficients and comparison with the one-loop formulas."""
import json, sys
import numpy as np
from oneloop import tick

f = sys.argv[1] if len(sys.argv) > 1 else "exact_main.json"
data = json.load(open(f))
ts = np.pi / 4
labels = ["y_H", "0", "-0.9y_H", "y_H-1", "y_H-2"]
Cs = sorted(set(r["C"] for r in data))
ms = sorted(set(r["m"] for r in data))
get = lambda C, m, lab, key: next(r[key] for r in data if r["C"] == C and r["m"] == m and r["label"] == lab and key in r)

def extrap(Cl, vals, deg):
    x = 1.0 / np.array(Cl)
    c = np.polyfit(x, vals, deg)
    return c[-1], c[-2]

print("Raw exact values (requested: C = 20, 40, m = 3)")
print(f"{'label':8s} {'y':>9s} {'R_cl':>9s} | {'C Var/<R>^2':>24s} | {'C(<R>/R_cl-1)':>24s}")
for lab in labels:
    y = next(r["y"] for r in data if r["label"] == lab)
    print(f"{lab:8s} {y: .5f} {np.sin(ts)/(np.cosh(y)-np.cos(ts)):9.6f} | "
          f"{get(20.0,3.0,lab,'v'):11.6f} {get(40.0,3.0,lab,'v'):11.6f}  | {get(20.0,3.0,lab,'a'):11.6f} {get(40.0,3.0,lab,'a'):11.6f}")

print("\n(i) C Var/<R>^2 : 1/C -> 0 extrapolation (polynomial in 1/C over C in", Cs, ")")
print(f"{'label':8s} {'m':>5s} " + " ".join(f"C={C:<7g}" for C in Cs) + "   deg2(C>=20)  deg3(all)   one-loop")
for lab in labels:
    y = next(r["y"] for r in data if r["label"] == lab)
    var1, a0_1, a1_1 = tick(ts, y)
    for m in [mm for mm in ms if mm > 1]:
        v = [get(C, m, lab, "v") for C in Cs]
        Csel = [C for C in Cs if C >= 20]
        e2 = extrap(Csel, [get(C, m, lab, "v") for C in Csel], 2)[0]
        e3 = extrap(Cs, v, 3)[0]
        print(f"{lab:8s} {m:5.2f} " + " ".join(f"{x:9.6f}" for x in v) + f"   {e2:9.6f}  {e3:9.6f}   {var1.real:9.6f}")

print("\n(ii) a(m,C) = C(<R>/R_cl - 1): extrapolate 1/C -> 0 at each m, then fit linear in m -> m=0")
for lab in labels:
    y = next(r["y"] for r in data if r["label"] == lab)
    var1, a0_1, a1_1 = tick(ts, y)
    ainf = []
    for m in ms:
        av = [get(C, m, lab, "a") for C in Cs]
        Csel = [C for C in Cs if C >= 20]
        e2 = extrap(Csel, [get(C, m, lab, "a") for C in Csel], 2)[0]
        e3 = extrap(Cs, av, 3)[0]
        ainf.append(e3)
        print(f"  {lab:8s} m={m:4.2f} " + " ".join(f"{x:10.6f}" for x in av) + f"  -> {e2:10.6f} {e3:10.6f}"
              f"   one-loop a0+a1 m = {(a0_1 + a1_1*m).real:10.6f}")
    c1 = np.polyfit(ms, ainf, 1); c2 = np.polyfit(ms, ainf, 2)
    print(f"  {lab:8s} linear fit: a(m->0) = {c1[1]:.6f}, slope {c1[0]:.6f};  quad fit: a0={c2[2]:.6f}, "
          f"slope {c2[1]:.6f}, curvature {c2[0]:.1e};  one-loop: a0 = {a0_1.real:.6f}, slope {a1_1.real:.6f}")

print("\nDirect double extrapolation at the requested C=20,40 only (m = 0.75,1.5,3; linear in 1/C, linear in m):")
for lab in labels:
    y = next(r["y"] for r in data if r["label"] == lab)
    ainf = []
    for m in [0.75, 1.5, 3.0]:
        a20, a40 = get(20.0, m, lab, "a"), get(40.0, m, lab, "a")
        ainf.append(2 * a40 - a20)
    c1 = np.polyfit([0.75, 1.5, 3.0], ainf, 1)
    v20, v40 = get(20.0, 3.0, lab, "v"), get(40.0, 3.0, lab, "v")
    print(f"  {lab:8s} a(m->0, C->inf) = {c1[1]:.5f}   C Var/R^2 (m=3, C->inf) = {2*v40 - v20:.5f}")
