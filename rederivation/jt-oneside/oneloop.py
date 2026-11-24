"""
One-loop (Gaussian Schwarzian fluctuation) prediction for the tick-rate moments.

phi(u) = u + eps(u), f = tan(phi/2), action I = (C/2) int (eps''^2 - eps'^2) du + ...
<eps(u1) eps(u2)> = G(u1-u2) = (1/2piC) sum_{|n|>=2} e^{in u}/(n^2(n^2-1))
                  = (1/2piC) [ -(u-pi)^2/2 + (u-pi) sin u + a + b cos u ],  0 <= Re u <= 2pi,
with a = 1 + pi^2/6, b = 5/2 (the values that remove n = 0, +-1).
Complex boundary points are handled by analytic continuation in the Euclidean ordering
(+ : -ts) < (y : iy) < (- : +ts)   (all pair separations have Re in (0, 2pi)).

L_ij = log B_ij = log 4 sin^2((phi_j - phi_i)/2) - log phi'_i - log phi'_j
d1 L = cot(u_ji/2) (e_j - e_i) - e'_i - e'_j
d2 L = -(1/4) csc^2(u_ji/2) (e_j - e_i)^2 + (e'_i^2 + e'_j^2)/2
log R = (L_{+-} - L_{+y} - L_{y-})/2
C Var R / R^2       -> C <(d1 logR)^2>
C(<R>_m/R_cl - 1)   -> C [ <d2 logR> + <(d1 logR)^2>/2 - m <d1 logR d1 L_{+-}> ]
two-point check:  C(<B^{-m}>/B_cl^{-m} - 1) -> C[ -m <d2 L> + m^2/2 <(d1 L)^2> ]
"""
import numpy as np

A0, B0 = 1 + np.pi ** 2 / 6, 2.5

def g(u, a=A0, b=B0, c=0.0):
    return -(u - np.pi) ** 2 / 2 + (u - np.pi) * np.sin(u) + a + b * np.cos(u) + c * np.sin(u)

def gp(u, a=A0, b=B0, c=0.0):
    return -(u - np.pi) + np.sin(u) + (u - np.pi) * np.cos(u) - b * np.sin(u) + c * np.cos(u)

def gpp(u, a=A0, b=B0, c=0.0):
    return -1 + 2 * np.cos(u) - (u - np.pi) * np.sin(u) - b * np.cos(u) - c * np.sin(u)

def check_series():
    n = np.arange(2, 200001)
    for u in [0.3, 1.7, 4.0]:
        sser = 2 * np.sum(np.cos(n * u) / (n ** 2 * (n ** 2 - 1.0)))
        print(f"  u={u}: closed g={g(u):.12f} series={sser:.12f}")

class OneLoop:
    """variables: index 2*P + d, P = point, d = 0 (eps) or 1 (eps')"""
    def __init__(self, pts, kw=None):
        self.u = pts            # ordered list of complex boundary positions (Euclidean order)
        self.kw = kw or {}
        n = len(pts)
        Cov = np.zeros((2 * n, 2 * n), dtype=complex)
        for i in range(n):
            for j in range(n):
                if i == j:
                    Cov[2*i, 2*i] = g(0.0, **self.kw)
                    Cov[2*i, 2*i+1] = Cov[2*i+1, 2*i] = 0.0
                    Cov[2*i+1, 2*i+1] = -gpp(0.0, **self.kw)
                else:
                    later, earlier = (i, j) if i > j else (j, i)
                    d = pts[later] - pts[earlier]
                    G, Gp, Gpp = g(d, **self.kw), gp(d, **self.kw), gpp(d, **self.kw)
                    # <eps_later eps_earlier> = G(d); d/du_later -> +, d/du_earlier -> -
                    Cov[2*later, 2*earlier] = G
                    Cov[2*later+1, 2*earlier] = Gp
                    Cov[2*later, 2*earlier+1] = -Gp
                    Cov[2*later+1, 2*earlier+1] = -Gpp
        Cov = np.where(np.abs(Cov) > 0, Cov, Cov.T)  # fill the transposed entries (symmetric covariance)
        self.Cov = Cov / (2 * np.pi)   # times 1/C, i.e. C*Cov

    def d1L(self, i, j):  # i earlier, j later
        v = np.zeros(self.Cov.shape[0], dtype=complex)
        x = self.u[j] - self.u[i]
        ct = 1 / np.tan(x / 2)
        v[2*j] += ct; v[2*i] -= ct
        v[2*i+1] -= 1; v[2*j+1] -= 1
        return v

    def d2L(self, i, j):  # <d2 L_ij>, times C
        x = self.u[j] - self.u[i]
        K = self.Cov
        e12sq = K[2*j, 2*j] + K[2*i, 2*i] - 2 * K[2*j, 2*i]
        return -0.25 / np.sin(x / 2) ** 2 * e12sq + 0.5 * (K[2*i+1, 2*i+1] + K[2*j+1, 2*j+1])

    def pair(self, v, w):
        return v @ self.Cov @ w

def tick(ts, y, m=0.0, kw=None):
    O = OneLoop([-ts, 1j * y, ts], kw)  # 0:+  1:y  2:-
    v = 0.5 * (O.d1L(0, 2) - O.d1L(0, 1) - O.d1L(1, 2))
    d2 = 0.5 * (O.d2L(0, 2) - O.d2L(0, 1) - O.d2L(1, 2))
    var = O.pair(v, v)
    a0 = d2 + 0.5 * var
    a1 = -O.pair(v, O.d1L(0, 2))
    return var, a0, a1

def twopt(ts, m, kw=None):
    O = OneLoop([-ts, ts], kw)
    v = O.d1L(0, 1)
    return -m * O.d2L(0, 1) + 0.5 * m ** 2 * O.pair(v, v)

if __name__ == "__main__":
    print("propagator closed form vs Fourier series:")
    check_series()
    ts = np.pi / 4
    yH = np.log(np.cos(ts))
    ys = [yH, 0.0, -0.9 * yH, yH - 1, yH - 2]
    names = ["y_H", "0", "-0.9y_H", "y_H-1", "y_H-2"]
    print("\none-loop two-point coefficient C(<B^-m>/B_cl^-m - 1) at u = 2ts:")
    for m in [0.75, 1.5, 3.0]:
        print(f"  m={m}: {twopt(ts, m).real:.10f}   (imag {twopt(ts, m).imag:.1e})")
    print("\none-loop tick-rate coefficients (C -> infinity):")
    print(f"{'label':8s} {'y':>9s} {'C Var/R^2':>14s} {'a0 = C(<R>/Rcl-1)|m=0':>24s} {'slope a1 (per unit m)':>22s}")
    for nm, y in zip(names, ys):
        var, a0, a1 = tick(ts, y)
        print(f"{nm:8s} {y: .5f} {var.real:14.10f} {a0.real:24.10f} {a1.real:22.10f}   imag: {abs(var.imag):.0e} {abs(a0.imag):.0e} {abs(a1.imag):.0e}")
    print("\ngauge test: change zero-mode constants a (const) and b (cos u) in the propagator (must not change invariant results)")
    for kw in [dict(a=0.0, b=0.0), dict(a=7.0, b=-3.0)]:
        out = [tick(ts, y, kw=kw) for y in ys]
        print(" ", kw, " var:", [f"{o[0].real:.6f}" for o in out], " a0:", [f"{o[1].real:.6f}" for o in out],
              " a1:", [f"{o[2].real:.6f}" for o in out], " 2pt m=3:", f"{twopt(ts, 3.0, kw=kw).real:.6f}")
