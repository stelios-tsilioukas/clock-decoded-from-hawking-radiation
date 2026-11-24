import numpy as np, json
from core import *
k1=20; h=0.25
p=physical_map(k1,np.pi/4,2.)
W=json.load(open('win_k20_De1.0_ee0.0125.json'))
g={k:grid(lo-1,hi+1,h) for k,(lo,hi) in W.items()}
s=Setup(p['C'],p['ts'],grids=g)
y=-1/p['kap']; sb=s.sb; ee=s.ee; De=s.De
D, hD = s.grids['D']; S, hS = s.grids['S']; E, hE = s.grids['E']
lD = logmu(D) - s.tD * D ** 2 + np.log(hD)
LIS = [s.logItri(a, y, D, S) for a in (0, 1)]
lgcS = logg(s.Dchi, D[:, None], S[None, :]); lFS = lse(lD[:, None] + lgcS, 0)
lge = logg(De, S[:, None], E[None, :])
lP = lse((logmu(S) + np.log(hS) - sb * (2 * s.ts - 2 * ee) * S ** 2 + lFS)[:, None] + lge, 0)
lE = logmu(E) + np.log(hE) - 2 * sb * ee * E ** 2 - lP
lws = logmu(S) + np.log(hS) - sb * (s.ts + s.eps / 2 - 2 * ee - 1j * y) * S ** 2
u = lws[:, None] + lge
LS=[lse(lD[:,None]+LIS[a],0) for a in (0,1)]
def contract(Q):
    R = np.empty((len(S), len(E)), complex)
    for i0 in range(0, len(S), 16):
        R[i0:i0 + 16] = lse(Q[i0:i0 + 16, :, None] + np.conj(u)[None, :, :], 1)
    return lse((u + R + lE[None, :]).ravel(), 0)
res={(a,b):contract(LS[a][:,None]+np.conj(LS[b])[None,:]) for a in (0,1) for b in (0,1)}
print('factorized Q via hd machinery', readout(res,0.2))
print('dece', readout(all_gluings(s,y,('dece',))['dece'],0.2))
# hd diagnostics: check N and M on diagonal
hg=0.5*lgcS
lN = lse(lD[:, None, None] + hg[:, :, None] + hg[:, None, :], 0)
print('N diag - F:', np.max(abs(np.diag(lN)-lFS)))
# correlation coefficient of the D-sharing: |M_00(sk,sb)|^2/(M(sk,sk)M(sb,sb))
