"""
Step 2 symbolic groundwork (conventions fixed here and used throughout Step 2).
Poincare AdS2:  a = t - z (ingoing label), b = t + z (outgoing label), boundary z=0 (a=b=t).
  ds^2 = -e^{2w} da db,  e^{2w} = 4/(b-a)^2.
JT equations (T_ab = 0, conformal matter; anomaly constant absorbed into phi_0):
  (E_aa)  -e^{2w} d_a(e^{-2w} d_a phi) = 8 pi G T_aa(a)
  (E_bb)  -e^{2w} d_b(e^{-2w} d_b phi) = 0           (outgoing modes in Poincare vacuum)
  (E_ab)  2 d_a d_b phi + e^{2w} phi = 0
"""
import sympy as sp
a, b, s, A, B, C, lam = sp.symbols('a b s alpha beta gamma lambda', real=True)
T = sp.Function('T')
e2w = 4/(b-a)**2
def Eaa(phi): return sp.simplify(-e2w*sp.diff(phi.diff(a)/e2w, a))
def Ebb(phi): return sp.simplify(-e2w*sp.diff(phi.diff(b)/e2w, b))
def Eab(phi): return sp.simplify(2*sp.diff(phi, a, b) + e2w*phi)

vac = (A + B*(a+b) + C*a*b)/(b-a)
print("vacuum family:  E_aa =", Eaa(vac), "  E_bb =", Ebb(vac), "  E_ab =", Eab(vac))

# source ansatz: phi_src = -lam * Int_0^a ds T(s) (a-s)(b-s) / (b-a)
src = -lam*sp.Integral(T(s)*(a-s)*(b-s), (s, 0, a))/(b-a)
print("source:  E_aa =", sp.simplify(Eaa(src).doit()), "   (want 8 pi G T(a))")
print("         E_bb =", sp.simplify(Ebb(src).doit()), "  E_ab =", sp.simplify(Eab(src).doit()))

# boundary value: (b-a) phi at a=b=t  ->  alpha + 2 beta t + gamma t^2 - lam Int_0^t T(s)(t-s)^2 ds
# third t-derivative of that = -2 lam T(t)
t = sp.symbols('t', real=True)
bv = A + 2*B*t + C*t**2 - lam*sp.Integral(T(s)*(t-s)**2, (s, 0, t))
print("d^3/dt^3 boundary value =", sp.simplify(sp.diff(bv, t, 3).doit()))

# Schwarzian identity: t = f(y), g(t) := f'(f^{-1}(t)).  Claim: d^3 g/dt^3 = Sch(f,y)'/f'^... check numerically-free:
y = sp.symbols('y', real=True)
f = sp.Function('f')
fp = f(y).diff(y)
Dt = lambda expr: sp.diff(expr, y)/fp           # d/dt = (1/f') d/dy
g = fp
g3 = sp.simplify(Dt(Dt(Dt(g))))
sch = sp.diff(f(y), y, 3)/fp - sp.Rational(3, 2)*(sp.diff(f(y), y, 2)/fp)**2
print("d^3 g/dt^3 - (d/dy Sch)/f'^2 =", sp.simplify(g3 - sp.diff(sch, y)/fp**2))
