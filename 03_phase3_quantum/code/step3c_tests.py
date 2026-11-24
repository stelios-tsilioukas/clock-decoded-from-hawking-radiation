import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'03_phase3_quantum/code')
from step3c_nonperturbative import NP
# (1) FFT implementation of the double-trumpet kernel on Gaussians: exact  sqrt(b1 b2)/(2 pi (b1+b2))
kmax=40; n=2**16; k=np.linspace(-kmax,kmax,n,endpoint=False); dk=k[1]-k[0]
for b1,b2 in ((1.0,1.0),(0.3,2.0),(0.2+0.5j,0.2-0.5j)):
    F1=np.fft.fft(np.fft.ifftshift(np.exp(-b1*k**2)))*dk; F2=np.fft.fft(np.fft.ifftshift(np.exp(-b2*k**2)))*dk
    q=2*np.pi*np.fft.fftfreq(n,d=dk)
    val=(np.abs(q)*F1*F2[(-np.arange(n))%n]).sum()*(2*np.pi/(n*dk))/(8*np.pi**2)
    print('double trumpet check',b1,b2, val, np.sqrt(b1*b2)/(2*np.pi*(b1+b2)))
# (2) grid convergence of disk and wormhole at large Dx
for Nk,Ns in ((12001,241),(24001,481)):
    X=NP(2.0,np.pi/4,2.0,Nk=Nk,Ns=Ns); r=X.run([0.0,0.35,8.0,32.0,64.0])
    print(Nk,Ns,[ (x['Dx'], x['disk']/r[0]['disk'], x['worm']/r[0]['disk']) for x in r], X.ramp_slope/r[0]['disk'])
