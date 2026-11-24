"""
A1: classical cubic amplitude on f = tan(u/2) vs. geodesic proper time in AdS2 (Rindler, beta=2pi).

Independent checks:
 (a) arg A_cl(y) from the triangle of bilocals (complex B's, principal branch, continuous in y)
 (b) proper time of the meeting point of the ingoing null ray from boundary time y with the clock
     geodesic, obtained by an explicit construction: Rindler null ray -> straight line in embedding
     space -> root-find on the clock worldline (no use of the closed form).
 (c) |A_cl| vs R(y)^h with R = d tau/dy (numerical derivative of (b)) and closed form.
"""
import numpy as np
from scipy.optimize import brentq

np.set_printoptions(precision=16)

def B(u1, u2):
    f = lambda u: np.tan(u / 2)
    fp = lambda u: 0.5 / np.cos(u / 2) ** 2
    return (f(u1) - f(u2)) ** 2 / (fp(u1) * fp(u2))

def A_cl(y, ts, Delta, omega, h):
    wa = (h - omega) / 2          # (+, y)
    wb = (h + omega) / 2          # (y, -)
    wc = Delta + omega / 2 - h / 2  # (+, -)
    up, um, uy = -ts, ts, 1j * y
    Bp, Bm, Bpm = B(up, uy), B(uy, um), B(up, um)
    # principal branch; B_+ = conj(B_-), Re sin((ts+iy)/2)>0 so arg B in (-pi,pi): continuous
    return np.exp(-wa * np.log(Bp) - wb * np.log(Bm) - wc * np.log(Bpm + 0j))

# ---------- geometry: AdS2 embedding  -X0^2 - T^2 + X1^2 = -1 ----------
def ip(X, Y):  # <X,Y> = -X0Y0 - T T' + X1 X1'  (components ordered (X0, X1, T))
    return -X[0] * Y[0] + X[1] * Y[1] - X[2] * Y[2]

def clock(sig, ts):
    # analytic continuation of the Euclidean geodesic X1 = cos(ts) X0 joining u = +-ts
    return np.array([np.cos(sig) / np.sin(ts), np.cos(sig) / np.tan(ts), np.sin(sig)])

def rindler(rho, t):
    return np.array([np.cosh(rho), np.sinh(rho) * np.cosh(t), np.sinh(rho) * np.sinh(t)])

def ingoing_line(y):
    """ingoing future-directed null ray from boundary time y, as a straight line P + lam*n."""
    r1, r2 = 3.0, 2.0
    t = lambda r: y - np.log(np.tanh(r / 2))  # dt/drho = -1/sinh rho
    P1, P2 = rindler(r1, t(r1)), rindler(r2, t(r2))
    n = P2 - P1
    return P1, n

def outgoing_line(u):
    r1, r2 = 3.0, 2.0
    t = lambda r: u + np.log(np.tanh(r / 2))  # past-directed from boundary: outgoing ray arriving at u
    P1, P2 = rindler(r1, t(r1)), rindler(r2, t(r2))
    return P1, P2 - P1

def meet(line, ts, lo, hi):
    """clock proper time sigma where the clock crosses the null line (null plane <X - P, m> = 0)."""
    P, n = line
    # the null line lies in the plane spanned by P and n through origin? points X on line satisfy
    # <X, n> = <P, n> (=0 since n null and P.n = 0 for a null geodesic on hyperboloid). Use the
    # 2-plane condition: X on hyperboloid and on the line <=> X = P + lam n. Clock crosses where
    # det[X, P, n] = 0 (X in the plane through origin spanned by P,n) AND that plane meets hyperboloid
    # along the line only on the correct sheet; we check the residual afterwards.
    f = lambda s: np.linalg.det(np.array([clock(s, ts), P, n]))
    # the plane span(P,n) meets the clock twice (X and -X, sigma and sigma+pi); scan for all roots
    # and keep the one lying on the line P + lam n (not on its antipode -P + lam n)
    grid = np.linspace(lo, hi, 2001)
    vals = np.array([f(g) for g in grid])
    best = None
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            s = brentq(f, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15)
            d = clock(s, ts) - P
            lam = (d @ n) / (n @ n)
            res = np.linalg.norm(d - lam * n)
            if best is None or res < best[1]:
                best = (s, res, lam)
    s, res, lam = best
    return s, res, lam

if __name__ == "__main__":
    ts = np.pi / 4
    Delta, omega, h = 1.3, 0.7, 0.45
    print("weights: w_a=(h-w)/2, w_b=(h+w)/2, w_c=Delta+w/2-h/2")
    # check null-ness & geodesic
    P, n = ingoing_line(0.3)
    print("ray tangent null:", ip(n, n), " P on hyperboloid:", ip(P, P) + 1, " <P,n>:", ip(P, n))
    s = 0.37; dX = np.array([-np.sin(s) / np.sin(ts), -np.sin(s) / np.tan(ts), np.cos(s)])
    print("clock: on hyperboloid", ip(clock(s, ts), clock(s, ts)) + 1, " |dX/dsig|^2 =", ip(dX, dX))

    sig_h = np.pi / 2 - ts   # horizon X1 = T
    Xh = clock(sig_h, ts)
    print("horizon crossing sigma_h=pi/2-ts:", sig_h, "X1-T =", Xh[1] - Xh[2], " T=e^{y_H} ->",
          np.log(Xh[2]), " ln cos ts =", np.log(np.cos(ts)), " -2artanh tan^2(ts/2) =",
          -2 * np.arctanh(np.tan(ts / 2) ** 2))

    sig0, res0, _ = meet(ingoing_line(0.0), ts, 0.0, np.pi)
    print("y=0 event: sigma =", sig0, "(pi/2 =", np.pi / 2, ") residual", res0)

    print("\n   y        argA/omega              sigma(y)-sigma(0)        diff        tau_closed")
    ys = np.linspace(-4, 4, 17)
    maxd = 0
    for y in ys:
        A = A_cl(y, ts, Delta, omega, h)
        ph = np.angle(A) / omega
        sg, res, _ = meet(ingoing_line(y), ts, -np.pi / 2 + 1e-9, 3 * np.pi / 2 - 1e-9) if False else (None, None, None)
        # bracket: sigma(y) in (ts - pi/2, 3pi/2 - ts)
        sg, res, _ = meet(ingoing_line(y), ts, ts - np.pi / 2 + 1e-12, 3 * np.pi / 2 - ts - 1e-12)
        tc = 2 * np.arctan(np.tanh(y / 2) / np.tan(ts / 2))
        maxd = max(maxd, abs(ph - (sg - sig0)))
        print(f"{y:6.2f}  {ph: .16f}  {sg - sig0: .16f}  {ph - (sg - sig0): .1e}  {tc: .16f}")
    print("max |argA/omega - Delta sigma| =", maxd)

    print("\n R(y): |A(y)|/|A(0)| vs (R(y)/R(0))^h ; R numeric derivative of sigma(y) vs closed form")
    Rc = lambda y: np.sin(ts) / (np.cosh(y) - np.cos(ts))
    for y in [-2.3466, -1.3466, np.log(np.cos(ts)), 0.0, 0.3119, 1.5]:
        e = 1e-5
        sp = meet(ingoing_line(y + e), ts, ts - np.pi / 2 + 1e-12, 3 * np.pi / 2 - ts - 1e-12)[0]
        sm = meet(ingoing_line(y - e), ts, ts - np.pi / 2 + 1e-12, 3 * np.pi / 2 - ts - 1e-12)[0]
        Rnum = (sp - sm) / (2 * e)
        r1 = abs(A_cl(y, ts, Delta, omega, h)) / abs(A_cl(0.0, ts, Delta, omega, h))
        r2 = (Rc(y) / Rc(0)) ** h
        print(f"y={y: .4f}  R_num={Rnum:.12f}  R_closed={Rc(y):.12f}  |A|ratio={r1:.15f}  (R/R0)^h={r2:.15f}  rel={r1/r2-1:.1e}")

    yH = np.log(np.cos(ts))
    print("\n y_H =", yH, ";  Delta tau (horizon -> y=0 event) =", 0 - 2 * np.arctan(np.tanh(yH / 2) / np.tan(ts / 2)),
          " = ts =", ts)

    print("\n outgoing (radar) labels: clock sigma emitting a ray received at boundary time u")
    for u in [0, 1, 2, 4, 8, 12, 16]:
        so, res, _ = meet(outgoing_line(u), ts, ts - 3 * np.pi / 2 + 1e-12, np.pi / 2 - ts - 1e-12)
        print(f" u={u:3d}  sigma_out={so:.15f}  sigma_h - sigma_out = {sig_h - so:.3e}   closed tau(u)-pi/2 = "
              f"{2*np.arctan(np.tanh(u/2)/np.tan(ts/2)) - np.pi/2:.15f}")
