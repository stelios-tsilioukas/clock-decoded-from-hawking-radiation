"""Check: classical reading (eps->0) = (proper time at which the ingoing ray from boundary time y meets the clock) - pi/2.
Global AdS2: ds^2=(-dT^2+dsig^2)/cos^2 sig, right boundary sig=pi/2, tan(T/2)=tanh(y/2); clock: sin sig = cos(tau_s) cos T."""
import numpy as np
from scipy.optimize import brentq
from sw import R_cl
ts=np.pi/4
for y in [-1.0,-0.3466,0.0,0.4,1.5]:
    Tb=2*np.arctan(np.tanh(y/2))
    f=lambda T: np.cos(T-Tb)-np.cos(ts)*np.cos(T)      # ingoing ray sig = pi/2-(T-Tb)
    T=brentq(f,Tb+1e-12,Tb+np.pi-1e-9)
    sig=np.pi/2-(T-Tb)
    # proper time along X0=cos(tau)/sqrt(1-A^2), X1 = sin(tau) with X1 = sinT/cos sig, X0 = cosT/cos sig
    tau=np.arctan2(np.sin(T)/np.cos(sig), np.cos(T)/np.cos(sig)*np.sin(ts))
    print(y, tau-np.pi/2, R_cl(ts,y,0.0))
