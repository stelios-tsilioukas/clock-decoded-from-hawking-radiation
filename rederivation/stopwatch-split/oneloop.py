"""
One-loop (O(1/C)) Schwarzian perturbation theory for the stopwatch reading, fixed dimensions.

 F_ab[phi] = prod_i phi'(u_i)^{D_i} G_c(phi(u_1..u_4)),   phi(u) = u + e(u),   f = tan(phi/2)
 G_c = conformal 4-pt chi-block on the circle (chordal distances s_ij = 2 sin((phi_i-phi_j)/2)).
 <F>/F_cl = 1 + <d2 log F> + 1/2 <(d1 log F)^2> + O(1/C^2)
 Propagator <e(u)e(0)> = P(u) = (1/2piC)[-(u-pi)^2/2 + (u-pi) sin u + 1 + pi^2/6 + 5/2 cos u],  0<=Re u<=2pi
 (Euclidean-ordered continuation for complex u).
 reading_1loop = R_cl + Im[ Delta log F_01 ] / omega           (F_00, F_11 stay real)
Returns the coefficient  a = C * (reading_1loop - R_cl)  (C-independent).
"""
import mpmath as mp

mp.mp.dps = 30


def P(u):           # times C
    return (-(u - mp.pi) ** 2 / 2 + (u - mp.pi) * mp.sin(u) + 1 + mp.pi ** 2 / 6 + mp.mpf(5) / 2 * mp.cos(u)) / (2 * mp.pi)


def dP(u):
    return (-(u - mp.pi) + mp.sin(u) + (u - mp.pi) * mp.cos(u) - mp.mpf(5) / 2 * mp.sin(u)) / (2 * mp.pi)


def d2P(u):
    return (-1 + 2 * mp.cos(u) - (u - mp.pi) * mp.sin(u) - mp.mpf(5) / 2 * mp.cos(u)) / (2 * mp.pi)


def logGc(p, D1, D2, D3, D4, Dx):
    p1, p2, p3, p4 = p
    s = lambda a, b: 2 * mp.sin((a - b) / 2)
    L21, L43, L42, L41, L31 = [mp.log(s(*q)) for q in ((p2, p1), (p4, p3), (p4, p2), (p4, p1), (p3, p1))]
    lchi = L21 + L43 - L31 - L42
    D12, D34 = D1 - D2, D3 - D4
    return (-(D1 + D2) * L21 - (D3 + D4) * L43 + D12 * (L42 - L41) + D34 * (L41 - L31)
            + Dx * lchi + mp.log(mp.hyp2f1(Dx - D12, Dx + D34, 2 * Dx, mp.exp(lchi))))


def dlogF(us, Ds, Dx):
    """one-loop correction C*<Delta log F> for insertion points us (ordered by real part) and dims Ds"""
    f = lambda a, b, c, d: logGc((a, b, c, d), *Ds, Dx)
    n = 4
    g = [mp.diff(f, us, tuple(1 if k == i else 0 for k in range(n))) for i in range(n)]
    H = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            o = [0] * n
            o[i] += 1
            o[j] += 1
            H[i][j] = H[j][i] = mp.diff(f, us, tuple(o))

    def EE(i, j):     # <e_i e_j>
        if i == j:
            return P(0)
        a, b = (i, j) if i > j else (j, i)
        return P(us[a] - us[b])

    def EdE(i, j):    # <e'_i e_j>
        if i == j:
            return mp.mpf(0)
        return dP(us[i] - us[j]) if i > j else -dP(us[j] - us[i])

    def dEdE(i, j):   # <e'_i e'_j>
        if i == j:
            return -d2P(0)
        a, b = (i, j) if i > j else (j, i)
        return -d2P(us[a] - us[b])

    d2 = sum(-Ds[i] * dEdE(i, i) / 2 for i in range(n)) + sum(H[i][j] * EE(i, j) / 2 for i in range(n) for j in range(n))
    d1sq = sum(Ds[i] * Ds[j] * dEdE(i, j) + 2 * Ds[i] * g[j] * EdE(i, j) + g[i] * g[j] * EE(i, j)
               for i in range(n) for j in range(n))
    return d2 + d1sq / 2


def a_coeff(tau, y, eps, Delta, omega, h):
    tau, y, eps, Delta, omega, h = map(mp.mpf, (tau, y, eps, Delta, omega, h))
    us = (-tau, 1j * y - eps / 2, 1j * y + eps / 2, tau)
    Dx = Delta + omega / 2
    Ds = (Delta, h, h, Delta + omega)
    return float(mp.im(dlogF(us, Ds, Dx)) / omega)


def a_coeff_real_check(tau, y, eps, Delta, omega, h):
    """Im <Delta log F_00> should vanish (F_00 real)"""
    tau, y, eps, Delta, omega, h = map(mp.mpf, (tau, y, eps, Delta, omega, h))
    us = (-tau, 1j * y - eps / 2, 1j * y + eps / 2, tau)
    Dx = Delta + omega / 2
    return complex(dlogF(us, (Delta, h, h, Delta), Dx))


if __name__ == '__main__':
    import numpy as np
    # check propagator vs mode sum
    u = 1.3
    ms = sum(2 * np.cos(n * u) / (n ** 2 * (n ** 2 - 1)) for n in range(2, 200000)) / (2 * np.pi)
    print('P check', float(P(mp.mpf(u))), ms)
    print(a_coeff_real_check(np.pi / 4, -0.3, 0.05, 2, 0.2, 1))
    for D in [0, 1, 2, 4]:
        print(D, a_coeff(np.pi / 4, -0.3, 0.05, D + 1e-30 if D == 0 else D, 0.2, 1))
