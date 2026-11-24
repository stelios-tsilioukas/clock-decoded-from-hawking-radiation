"""At h,omega->0 the one-loop coefficient must be exactly linear in Delta: a = a_q + Delta*slope.
Compare slope with the classical temperature-shift slope b from the kappa1 map (m = Delta)."""
import numpy as np
from sw import phys_to_bare, yH_of, R_cl, bare_to_phys
from oneloop import a_coeff
d = phys_to_bare(20.0, np.pi/4, 2.0); C, k, tau = d['C'], d['kappa1'], d['tau']
y = yH_of(np.pi/4)/k; eb = 0.05/k
A = [a_coeff(tau, y, eb, D, 1e-5, 1e-5) for D in (1e-9, 1, 2, 3)]
print('a(Delta=0,1,2,3) =', A, ' slopes', np.diff(A))
Cb=1e6; kb=bare_to_phys(Cb,tau,1.0)['kappa1']; print('b (kappa map) =', Cb*(R_cl(kb*tau,kb*y,kb*eb)-R_cl(tau,y,eb)))
# protocol kinematics (tau=pi/4, y=-0.3): Delta-dependence at h=1, omega=0.2 is not linear (block terms ~h^2/Delta)
