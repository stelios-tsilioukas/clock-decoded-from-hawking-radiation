"""
Checks of the planar derivation on single samples:
  (1) block-diagonal expectations g_e(s) = [(s+GG^dag)^{-1}]_e, x_g(s) = s [C (s+tau)^{-1} C^dag]_g, m_alpha
      measured in a sample vs. the Dyson-equation solution;
  (2) 'factorised' own/unprojected formula evaluated with the sample's own block averages vs. the direct
      eps-odd Petz computation in the same sample (tests the chord/factorisation structure of the derivation,
      independently of the Dyson solver).
usage: python check_factorization.py REGIME k seed
"""
import sys, numpy as np
from model import Setup, petz_quantities
from planar import Planar

reg = sys.argv[1]; k = int(sys.argv[2]); seed = int(sys.argv[3])
S = Setup(reg); rng = np.random.default_rng(seed); L = S.L; eps = S.p['eps']; y = S.p['y']
m = S.draw_matter(rng)
T = S.brane(rng, k)
X0 = S.Xmat(m['O']); XZ = S.Xmat(m['Z'])
Psi0 = X0 @ T; N0 = np.vdot(Psi0, Psi0).real; Psi0 /= np.sqrt(N0)
PZ = XZ @ T; N1 = np.vdot(PZ, PZ).real; PsiZ = PZ / np.sqrt(N1)
tau = (Psi0.conj().T @ Psi0 + PsiZ.conj().T @ PsiZ) / 2
lam, U = np.linalg.eigh(tau)
G = np.vstack([Psi0, PsiZ]) / np.sqrt(2)
GU = (G @ U) / np.sqrt(lam)[None, :]                     # normalised eigvecs of GG^dag
blk2 = np.concatenate([S.blk, S.blk + 5])
Vm = np.array([np.mean(np.abs(GU[blk2 == e]) ** 2, 0) for e in range(10)]) # (10, r)
pi_e = Vm.sum(1)
CU = T @ U
Bm = np.array([np.mean(np.abs(CU[S.blk == g]) ** 2, 0) for g in range(5)])  # (5, r)
Mdiag = np.real(np.einsum('ij,j,ij->i', Psi0 @ U, lam ** -0.5, (Psi0 @ U).conj()))
m_mc = np.array([Mdiag[S.blk == a].mean() for a in range(5)])

P = Planar(reg, k); I = P.integrals()
vs = P.vs; s = np.exp(vs)
g_mc = (1 - pi_e)[None, :] / s[:, None] + (1.0 / (s[:, None] + lam[None, :])) @ Vm.T
x_mc = (s[:, None] / (s[:, None] + lam[None, :])) @ Bm.T
print(f'== {reg} k={k} seed={seed}')
for v0 in (-20, -10, -5, 0):
    i = np.argmin(np.abs(vs - v0))
    print(f' s=e^{v0}: s*g(code0) MC', np.round(s[i] * g_mc[i, :5], 4), ' Dyson', np.round(s[i] * P.G[i, :5], 4))
    print(f'          s*g(code1) MC', np.round(s[i] * g_mc[i, 5:], 4), ' Dyson', np.round(s[i] * P.G[i, 5:], 4))
    print(f'          x/k        MC', np.round(x_mc[i] / k, 4), ' Dyson', np.round(P.X[i] / k, 4))
print(' m_alpha MC   ', np.round(m_mc, 5)); print(' m_alpha Dyson', np.round(I['m'], 5))
w = s ** 0.5 / np.pi * (vs[1] - vs[0]); w[0] *= .5; w[-1] *= .5
T_mc = np.einsum('i,ia,ig->ag', w, g_mc[:, 5:], x_mc)
Tu_mc = np.einsum('i,ia,ib,ig->abg', w * s, g_mc[:, 5:], g_mc[:, :5], x_mc)
wt = P.wtensor(y)
nrm = 1 / np.sqrt(N0 * N1)   # sample normalisations (planar ones are P.N)
fac_own = eps / 2 * nrm * np.sum(wt * m_mc[:, None, None] * T_mc[:, None, :])
fac_unp = eps / 2 * nrm * np.sum(wt * m_mc[:, None, None] * Tu_mc)
# direct computation in the same sample
Wo = S.Xmat(S.W_of(m['V'], m['Ot'])); Wu = S.Xmat(S.W_of(m['V'], m['O']))
res = {}
for name, Wx, O in (('own', Wo, m['Ot']), ('unproj', Wu, m['O'])):
    Phi = S.Ymat(m['V'], O, y) @ T / np.sqrt(N0)
    vals = []
    for sg in (1, -1):
        Psi1 = PZ + sg * eps * (Wx @ T); Psi1 /= np.linalg.norm(Psi1)
        vals.append(petz_quantities(Psi0, Psi1, [Phi])[0][0])
    res[name] = (vals[0] - vals[1]) / 2
r = P.predict(y, eps, I)
print(' own   : direct', np.round(res['own'], 6), ' factorised(sample blocks)', np.round(fac_own, 6), ' planar', np.round(r['own'], 6))
print(' unproj: direct', np.round(res['unproj'], 6), ' factorised(sample blocks)', np.round(fac_unp, 6), ' planar', np.round(r['unproj'], 6))
print(' sample N0,N1 vs planar', N0, N1, P.N)
