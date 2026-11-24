import numpy as np
from sw import *
from oneloop import a_coeff
tau=np.pi/4
# classical backreaction slope: lim C*(R_cl(kappa1 ...)-R_cl(1...)) at fixed y
y=-0.3
for C in [1e3,1e4,1e5]:
    for D in [1,2]:
        d=bare_to_phys(C,tau,D); k=d['kappa1']
        print(C,D,C*(k-1), C*(R_cl(k*tau,k*y,k*0.05)-R_cl(tau,y,0.05)))
aD=[a_coeff(tau,y,0.05,D,0.2,1) for D in (1e-12,1,2)]
print('PT a(0),a(1),a(2)',aD, 'slope',aD[1]-aD[0], aD[2]-aD[1])
# small omega: is slope omega-dependent?
for om in [0.2,0.02]:
    aa=[a_coeff(tau,y,0.05,D,om,1) for D in (1e-12,1)]
    print('omega',om, aa, aa[1]-aa[0])
