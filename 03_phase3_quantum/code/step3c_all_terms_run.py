import numpy as np, sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'03_phase3_quantum/code')
from step3c_all_terms import AllTerms
C=float(sys.argv[1]); p=float(sys.argv[2]); m=float(sys.argv[3]); Dxs=[float(x) for x in sys.argv[4].split(',')]; tag=sys.argv[5]
X=AllTerms(C,np.pi/4,m,p=p)
Nn=X.norm(); dn=(Nn['ab']+Nn['Ga']+Nn['Gb'])/Nn['disk']
c2=(4*np.pi**2)**2; rows=[]; M0=None
for Dx in Dxs:
    T=X.run(Dx); M=T['disk']; M0=M0 or M; tot=sum(v for k,v in T.items() if k!='disk')/M
    rows.append(dict(Dx=Dx, disk_rel0=M/M0, **{k:v/M0 for k,v in T.items() if k!='disk'}, dMp_over_Mp=tot, dM0_over_M0=dn, c=c2*(tot-dn), c_bc_only=c2*T['bc']/M))
    print(f"C={C} p={p} m={m} Dx={Dx:7.4f}: disk/disk0={M/M0:.3e} | abs terms (units disk0): "+" ".join(f"{k}={v/M0:+.2e}" for k,v in T.items() if k!='disk')+f" | c={rows[-1]['c']:+.4e}",flush=True)
json.dump(dict(C=C,p=p,m=m,norm_rel=dn,rows=rows),open(f'03_phase3_quantum/results/step3c_all_terms_C{C:g}_p{p:g}_m{m:g}_{tag}.json','w'),indent=1,default=float)
