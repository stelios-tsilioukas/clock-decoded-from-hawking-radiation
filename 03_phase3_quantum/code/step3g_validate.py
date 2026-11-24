import numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'03_phase3_quantum/code')
import step3g_decoding as D, step3b_exact_rate as old
N=1e4
print("microcanonical c1 (PSSY 3.43: k/N for k<<N, 1 - N/4k for k>>N):")
for r in (1e-3,1e-2,0.1,0.5,1,2,10,100,1000):
    k=r*N; c=D.c1_micro(k,N); print(f"  k/N={r:8.3g}: c1={c:.6f}   small-k {k/N:.6f}  large-k {1-N/(4*k):.6f}")
# moments without filter must reproduce Step 3b
X=D.Decoder(10.0,np.pi/4,2.0,S0=8.0)
out,c1,Sbh,sk=X.decoded_moments([-0.347,0.0],[1,2],logk=200.0)   # huge k: G=1
Xo=old.Exact(10.0)
for j,dy in enumerate((-0.347,0.0)):
    ref=Xo.moments(np.pi/4,dy,2.0,[1,2]); print("dy",dy,"phys",out[1][j][0].real,out[2][j][0].real,"| dec",out[1][j][1].real,out[2][j][1].real,"| 3b",ref[1].real,ref[2].real)
print("c1 at huge k:",c1," S_BH(clock side) - S0 =",Sbh-8.0, " 2 pi C =",2*np.pi*10)
