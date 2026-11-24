#!/usr/bin/env python3
"""Compare exact log(Z_T/Z_C) (results_main.json, results_C160.json) with
   probe limit 4 D log tan(tau/2) and with the derived O(1/C) correction
   [D^2 (Q(pi-tau)-Q(tau)) + D (L(pi-tau)-L(tau))]/C   (mu = infinity)."""
import json, numpy as np

PI = np.pi


def Q(s):
    T = 1 / np.tan(s / 2); l = 2 * s
    return 4 * T * (1 - l / PI) + T**2 * l**2 / (2 * PI) - l * T**2


def L(s):
    T = 1 / np.tan(s / 2); l = 2 * s
    return 2 * T * (1 - l / (2 * PI)) - l * (1 + T**2) / 2 + (1 + T**2) * l**2 / (8 * PI)


def corr(D, tau, C):
    return (D**2 * (Q(PI - tau) - Q(tau)) + D * (L(PI - tau) - L(tau))) / C


rows = []
for f in ["results_main.json", "results_C160.json"]:
    rows += json.load(open(f"rederivation/threading/{f}"))

print("| C | Delta | tau/pi | exact log(Z_T/Z_C) | probe 4D log tan(tau/2) | exact-probe | probe+1/C | exact-(probe+1/C) |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    tau = r["tau_over_pi"] * PI
    c1 = corr(r["D"], tau, r["C"])
    ex, pr = r["logratio"], r["probe"]
    print(f"| {r['C']} | {r['D']} | {r['tau_over_pi']} | {ex:.6f} | {pr:.6f} | {ex-pr:+.6f} | {pr+c1:.6f} | {ex-pr-c1:+.2e} |")
