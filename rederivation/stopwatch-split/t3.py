import numpy as np, time
from sw import *
from oneloop import a_coeff
tau=np.pi/4; y=-0.3
print('PT a =', a_coeff(tau,y,0.05,2,0.2,1))
for C in [10,20,40,80,160]:
    t=time.time()
    r=exact_F(C,tau,y,0.05,2.0,0.2,1.0,dk=0.1 if C<100 else 0.15)
    print(C, C*(r['reading']-R_cl(tau,y,0.05)), time.time()-t, flush=True)
