import numpy as np, time
from sw import *
for C in [10,20]:
    d=bare_to_phys(C,np.pi/4,2.0); print(C, d['k1'], d['taup'], d['kappa1'])
    for dk in [0.2,0.1,0.05]:
        t=time.time()
        r=exact_F(C,np.pi/4,-0.3,0.05,2.0,0.2,1.0,dk=dk)
        print(dk, r['N'],r['NB'], r['reading'], r['absV'], r['imF00'], time.time()-t)
    print('cl', R_cl(np.pi/4,-0.3,0.05))
