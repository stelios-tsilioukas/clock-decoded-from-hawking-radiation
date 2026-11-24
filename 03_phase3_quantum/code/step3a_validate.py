"""Validation of the propagator usage in step3a_oneloop.py against Maldacena-Stanford-Yang eqs (4.30) and (4.32),
and the split of the contour-ordered variance into a symmetrised (Keldysh) part and a commutator part."""
import numpy as np, sys, json
sys.path.insert(0, '03_phase3_quantum/code')
from step3a_oneloop import G, oneloop, points, PAIRS, ORDER
C = 1.0
def dG(u, h=1e-5): return (G(u + h, C) - G(u - h, C))/(2*h)
def d2G(u, h=1e-4): return (G(u + h, C) - 2*G(u, C) + G(u - h, C))/h**2
def ee(ua, ub):   return G(abs(ua - ub), C)
def epe(ua, ub):  return dG(abs(ua - ub))*np.sign(ua - ub)          # <eps'(a) eps(b)>
def epep(ua, ub): return -d2G(abs(ua - ub))                          # <eps'(a) eps'(b)>
def BB(u1, u2, u3, u4):
    """<B(u1,u2) B(u3,u4)>, B = eps'_1 + eps'_2 - (eps_1 - eps_2)/tan(u12/2), Euclidean-ordered correlators."""
    t12, t34 = 1/np.tan((u1 - u2)/2), 1/np.tan((u3 - u4)/2)
    X = [(u1, 'p', 1.0), (u2, 'p', 1.0), (u1, 'e', -t12), (u2, 'e', t12)]
    Y = [(u3, 'p', 1.0), (u4, 'p', 1.0), (u3, 'e', -t34), (u4, 'e', t34)]
    tot = 0.0
    for ua, ka, ca in X:
        for ub, kb, cb in Y:
            if ka == 'p' and kb == 'p': v = epep(ua, ub)
            elif ka == 'p': v = epe(ua, ub)
            elif kb == 'p': v = epe(ub, ua)
            else: v = ee(ua, ub)
            tot += ca*cb*v
    return tot
E = lambda u: -2 + u/np.tan(u/2)
out = {}
u1, u2, u3, u4 = 2.6, 1.9, 1.1, 0.3                                   # ordered u4<u3<u2<u1
out['MSY (4.30) ordered: numeric - formula'] = float(BB(u1, u2, u3, u4) - E(u1 - u2)*E(u3 - u4)/(2*np.pi*C))
u1, u3, u2, u4 = 2.6, 1.9, 1.1, 0.3                                   # crossed u4<u2<u3<u1
f432 = (E(u1 - u2)*E(u3 - u4) + 2*np.pi*(np.sin((u1 - u2 + u3 - u4)/2) - np.sin((u1 + u2 - u3 - u4)/2))
        /(np.sin((u1 - u2)/2)*np.sin((u3 - u4)/2)) + 2*np.pi*(u2 - u3)/(np.tan((u1 - u2)/2)*np.tan((u3 - u4)/2)))/(2*np.pi*C)
out['MSY (4.32) crossed: numeric - formula'] = float(BB(u1, u2, u3, u4) - f432)

# ---- Keldysh / commutator split for the clock variance: swap the order of the two real-time insertions
import step3a_oneloop as S
def V1_order(ts, y1, y2, order):
    S.ORDER[:] = order
    v = oneloop(ts, y1, y2)[0]; S.ORDER[:] = ['+', '1', '2', '-']; return v
ts, y1, y2 = np.pi/4, -1.0, 0.2
Va = V1_order(ts, y1, y2, ['+', '1', '2', '-']); Vb = V1_order(ts, y1, y2, ['+', '2', '1', '-'])
out['time-ordered  -V1'] = [float((-Va).real), float((-Va).imag)]
out['anti-ordered  -V1'] = [float((-Vb).real), float((-Vb).imag)]
out['symmetrised variance (Keldysh)'] = [float((-(Va + Vb)/2).real), float((-(Va + Vb)/2).imag)]
print(json.dumps(out, indent=1))
json.dump(out, open('03_phase3_quantum/results/step3a_validate.json', 'w'), indent=1)
