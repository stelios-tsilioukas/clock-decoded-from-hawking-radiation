import numpy as np, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3e_matter_clock as M, step3d_backreaction as B
out={}
for tag,ts in (('pi4',np.pi/4),('3pi8',3*np.pi/8)):
    dH=B.dy_horizon(ts); labs=[('y_H-2',dH-2),('y_H-1',dH-1),('horizon',dH),('phi=0',0.0),('0.9 inner',-0.9*dH)]
    print("tau_s =",tag)
    rows=[]
    for nm,dy in labs:
        c=M.lnZhat_coeffs_bar(ts,dy)
        rows.append(dict(label=nm,dy=dy,**{k:[v.real,v.imag] for k,v in c.items()}))
        print(f"  {nm:9s}: "+"  ".join(f"{k}={v.real:+.4f}{v.imag:+.4f}i" for k,v in c.items()))
    # check tau_phys = tau_cl - tau_cl(0)
    print("  tau_phys check:", max(abs(M.tau_phys(ts,d)-(M.tau_cl(ts,d)-M.tau_cl(ts,0))) for _,d in labs))
    out[tag]=rows
json.dump(out,open('03_phase3_quantum/results/step3e_oneloop_coeffs.json','w'),indent=1)
