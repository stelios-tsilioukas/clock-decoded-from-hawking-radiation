"""basic checks: Bartlett, Petz normalisation, eigh-vs-SVD precision, timing"""
import time, numpy as np
from model import Setup, petz_quantities
S = Setup('thread'); rng = np.random.default_rng(1)
L = S.L
# Bartlett: E[T T^dag] = k I
k = 1600
acc = np.zeros((L, L), complex)
for _ in range(20):
    T = S.brane(rng, k); acc += T @ T.conj().T
acc /= 20
print('Bartlett: mean diag/k', np.mean(np.diag(acc).real)/k, ' offdiag rms/k', np.sqrt(np.mean(np.abs(acc-np.diag(np.diag(acc)))**2))/k, '(expect ~ 1/sqrt(20k)=%.4f)'%(1/np.sqrt(20*k)))
# eigenvalue spread of tau, Petz identity checks
m = S.draw_matter(rng)
for k in (48, 1600):
    T = S.brane(rng, k)
    P0 = S.Xmat(m['O']) @ T; P0 /= np.linalg.norm(P0)
    P1 = S.Xmat(m['Z']) @ T; P1 /= np.linalg.norm(P1)
    t = time.time()
    (r10, r01, r11), r00, (lam, U) = petz_quantities(P0, P1, [P0, P1, P1], want_eig=True)
    print('k', k, 'eigh time', time.time()-t, 'lam range', lam.min(), lam.max())
    # Petz: <Psi0|O(1)|Psi0> = rho00 + <Psi0|O(|1><1|)|Psi0>
    tau = (P0.conj().T@P0 + P1.conj().T@P1)/2
    # SVD-based polar
    G = np.vstack([P0, P1])/np.sqrt(2)
    t = time.time(); Uu, s, Vh = np.linalg.svd(G, full_matrices=False); print('  svd time', time.time()-t, 'smin^2', s.min()**2)
    Pp = Uu @ Vh
    Q0, Q1 = np.sqrt(2)*Pp[:L], np.sqrt(2)*Pp[L:]   # Psi_a tau^{-1/2}
    rho00_svd = 0.5*np.trace(P0.conj().T @ Q0 @ (P0.conj().T @ Q0))
    print('  rho00 eigh', r00, ' svd', rho00_svd)
    # O_R(1) on code = 1 : <Psi0|O(|0><0|)+O(|1><1|)|Psi0> should be 1
    Kt = 0.5*np.linalg.inv(np.linalg.cholesky(tau)).conj().T  # dummy
    lamh = lam**-0.5
    Ti = (U*lamh)@U.conj().T
    O_id_T = 0.5*Ti@(P0.conj().T@P0 + P1.conj().T@P1)@Ti
    print('  <Psi0|O(1)|Psi0> =', np.trace(P0.conj().T@P0@O_id_T).real, ' <Psi1|O(1)|Psi1> =', np.trace(P1.conj().T@P1@O_id_T).real)
