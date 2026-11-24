"""
Step 3a -- one-loop Schwarzian fluctuations of the clock's proper time (boundary protocol).

Conventions (Maldacena-Stanford-Yang 1606.01857, sec. 4.1): beta = 2 pi, f = tan(tau/2), tau(u) = u + eps(u),
  <eps(u) eps(0)> = (1/(2 pi C)) [ -(|u|-pi)^2/2 + (|u|-pi) sin|u| + a + b cos u ]            (their 4.28)
Here kappa = 1/2, S - S0 = 2 pi C, and the Euclidean displacement of the creation operator is delta = tau_s.
Contour (Euclidean part, Lorentzian time):  u_+ = -tau_s + i y_d (ket),  u_1 = i y_1,  u_2 = i y_2 (forward branch,
time ordered),  u_- = +tau_s + i y_d (bra).  Contour order: +, 1, 2, -.
  chi = s(+,2) s(-,1) / [ s(-,2) s(+,1) ],  s(a,b) = sin((u_a-u_b)/2);     Dtau = arg(chi) = -i log chi,  exp(-i omega Dtau) = chi^(-omega)
  log <chi^p> = p log chi_cl + p <d2 log chi> + (p^2/2) <(d1 log chi)^2> + O(1/C^2)
"""
import numpy as np, json, sys

def G(u, C, a=1 + np.pi**2/6, b=2.5):
    """contour-ordered propagator, argument = (later) - (earlier), Re u >= 0."""
    return (-(u - np.pi)**2/2 + (u - np.pi)*np.sin(u) + a + b*np.cos(u))/(2*np.pi*C)

PAIRS = [('+', '2', +1), ('-', '1', +1), ('-', '2', -1), ('+', '1', -1)]
ORDER = ['+', '1', '2', '-']

def points(ts, y1, y2, yd=0.0):
    return {'+': -ts + 1j*yd, '1': 1j*y1, '2': 1j*y2, '-': ts + 1j*yd}

def logchi_cl(ts, y1, y2):
    u = points(ts, y1, y2)
    return sum(s*np.log(np.sin((u[a] - u[b])/2)) for a, b, s in PAIRS)

def oneloop(ts, y1, y2, C=1.0, a=1 + np.pi**2/6, b=2.5):
    u = points(ts, y1, y2)
    c = {k: 0j for k in ORDER}
    for p, q, s in PAIRS:                                   # d1 log chi = sum_a c_a eps_a
        w = s/(2*np.tan((u[p] - u[q])/2)); c[p] += w; c[q] -= w
    def prop(p, q):                                         # contour-ordered <eps_p eps_q>
        if p == q: return G(0.0, C, a, b)
        i, j = ORDER.index(p), ORDER.index(q)
        early, late = (p, q) if i < j else (q, p)
        return G(u[late] - u[early], C, a, b)
    V1 = sum(c[p]*c[q]*prop(p, q) for p in ORDER for q in ORDER)
    # second order: d2 log chi = - sum_pairs s (eps_a - eps_b)^2 / (8 sin^2(u_ab/2))
    d2 = -sum(s*(2*G(0.0, C, a, b) - 2*prop(p, q))/(8*np.sin((u[p] - u[q])/2)**2) for p, q, s in PAIRS)
    return V1, d2

def dtau_cl(ts, y1, y2):
    """classical proper time between the two signals (closed form of the audit, kappa = 1/2)."""
    t = lambda y: 2*np.arctan2(np.cosh(-y/2)*np.sin(ts/2), np.sinh(-y/2)*np.cos(ts/2))
    return t(y2) - t(y1)

def K_bdy(ts, y1, y2, h=1e-6):
    """d Dtau / d ln kappa at fixed boundary data (y_i, delta): everything scales with kappa."""
    f = lambda lam: dtau_cl(lam*ts, lam*y1, lam*y2)
    return (f(1 + h) - f(1 - h))/(2*h)

if __name__ == "__main__":
    out = {'checks': {}, 'scan': []}
    # ---- sign convention and classical check
    ts, y1, y2 = np.pi/4, -1.2, 0.2
    u = points(ts, y1, y2)
    chi = np.prod([np.sin((u[a] - u[b])/2)**s for a, b, s in PAIRS])     # use chi itself: no log-branch ambiguity
    out['checks']['classical: |chi| - 1'] = float(abs(chi) - 1)
    out['checks']['classical: arg(chi) - Dtau(closed form)'] = float(np.angle(chi) - dtau_cl(ts, y1, y2))
    # ---- gauge invariance (independence of the zero-mode coefficients a, b)
    V1, d2 = oneloop(ts, y1, y2); V1b, d2b = oneloop(ts, y1, y2, a=-3.3, b=7.1)
    out['checks']['gauge: |V1(a,b) - V1(a\',b\')|'] = float(abs(V1 - V1b))
    out['checks']['gauge: |d2(a,b) - d2(a\',b\')|'] = float(abs(d2 - d2b))
    print(json.dumps(out['checks'], indent=1))
    # ---- scan.  With Dtau = sigma * (-i log chi), sigma = +-1 from the check above:
    #   <(d tau)^2> = -V1 ;   energy-fluctuation prediction (Step 1):  K^2/(S - S0) = K^2/(2 pi C)
    print("\n C=1 (S-S0 = 2 pi).   var = -V1;  E-part = K^2/(2 pi C);  rest = var - E-part;  shift = mean shift of Dtau")
    for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
        ylast = np.log(1/np.cos(ts)); yH = None
        for (y1, y2) in [(-3.0, -1.0), (-2.0, 0.0), (-1.0, 0.0), (-1.0, 0.8*ylast), (0.0, 0.8*ylast), (-0.5, 0.95*ylast)]:
            V1, d2 = oneloop(ts, y1, y2); var = -V1; K = K_bdy(ts, y1, y2); E = K**2/(2*np.pi)
            shift = d2/1j                                   # <d2 tau>, Dtau = arg(chi) = -i log chi
            row = dict(ts=ts, y1=y1, y2=y2, dtau=float(dtau_cl(ts, y1, y2)), var_re=float(var.real), var_im=float(var.imag),
                       K=float(K), E_part=float(E), rest_re=float(var.real - E), shift_re=float(shift.real), shift_im=float(shift.imag))
            out['scan'].append(row)
            print(f" ts={ts:.3f} y1={y1:+.2f} y2={y2:+.3f} Dtau={row['dtau']:.4f} | var = {var.real:+.5f} {var.imag:+.5f}i | "
                  f"K={K:+.4f} E-part={E:.5f} | rest={row['rest_re']:+.5f} | shift = {shift.real:+.5f} {shift.imag:+.5f}i")
    json.dump(out, open('03_phase3_quantum/results/step3a_oneloop.json', 'w'), indent=1)
