import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3d_backreaction as B
ts=np.pi/4
print("bare (C, m): exact sqrt<k0^2>, sqrt<ks^2> vs saddle C*x0, C*x1; exact energy identity")
for C,m in ((40,2),(10,2),(4,2),(2,2),(1,2),(0.5,2),(10,10)):
    e0,e1=B.energies(C,ts,m); x0,x1=B.saddle(m/C,ts)
    print(f"C={C} m={m}: {np.sqrt(e0):.4f} {np.sqrt(e1):.4f} | {C*x0:.4f} {C*x1:.4f} | <E1-E0>={e1-e0:.4f} saddle {C*C*(x1**2-x0**2):.4f}")
