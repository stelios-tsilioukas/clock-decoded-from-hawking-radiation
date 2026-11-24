import numpy as np, time
from sw import *
tau=np.pi/4
for C in [10,20,40,80]:
    d=bare_to_phys(C,tau,2.0); kap=d['kappa1']
    yH=yH_of(kap*tau)/kap
    for lab,y in [('H',yH),('0.9in',-0.9*yH)]:
        t=time.time()
        r=exact_F(C,tau,y,0.05,2.0,0.2,1.0,dk=0.1)
        r2=exact_F(C,tau,y,0.05,2.0,0.2,1.0,dk=0.1,kmax=C+12*np.sqrt(C)+20,kmaxB=C+12*np.sqrt(C)+200)
        c1=R_cl(tau,y,0.05); ck=R_cl(kap*tau,kap*y,kap*0.05)
        print(C,lab,y,r['reading'],r2['reading']-r['reading'],'cl1',c1,'clk',ck,'C*(ex-cl1)',C*(r['reading']-c1),'C*(ex-clk)',C*(r['reading']-ck),time.time()-t)
