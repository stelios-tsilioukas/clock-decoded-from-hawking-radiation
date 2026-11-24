"""
Random-matrix model (PSSY app. D style, with matter) for Petz decoding of an infalling clock.
Independent implementation for re-derivation task A4 (agent rmt-absorption).

Conventions (see report.md):
  * ket amplitude of brane state i:  sum_n C_{ni} f_n |n>,   f_n^2 = f2[block(n)]  (single power of f, see report)
  * states are L x k matrices Psi[n, i]  (B index n, R index i)
  * X_a = e^{-t_c H/2} O_a e^{-t_b H/2} F ,  Psi_a = X_a C / sqrt(N_a),  N_a = ||X_a C||_F^2
  * Y   = e^{-tau1 H} V e^{-tau2 H} O e^{-t_b H/2} F ,  Phi = Y C / sqrt(N_0)
  * Petz (Heisenberg) with sigma = (rho_0+rho_1)/2, normalised so O_R(1) = 1 on code:
        tau := (Psi_0^dag Psi_0 + Psi_1^dag Psi_1)/2   (= sigma^T, k x k)
        O_R(|1><0|)^T = (1/2) tau^{-1/2} Psi_0^dag Psi_1 tau^{-1/2}
        rho10 = <Phi|O_R(|1><0|)|Psi_0> = (1/2) Tr( Phi^dag Psi_0 tau^{-1/2} Psi_0^dag Psi_1 tau^{-1/2} )
        rho00 = (1/2) Tr( Psi_0^dag Psi_0 tau^{-1/2} Psi_0^dag Psi_0 tau^{-1/2} )
  * for k >= L:  C = T Q (Bartlett, T lower triangular L x L, Q isometry) and all traces reduce to L x L.
"""
import numpy as np

BASE = dict(Ls=[30, 120, 200, 250, 300], E=[0.0, 0.7, 1.4, 2.1, 2.8], y=0.9, eps=0.05,
            f2=[1.0, 0.8, 0.65, 0.5, 0.4])
REGIMES = {
    'thread':  dict(BASE, tc=8.0, tb=0.05, a=0.4),
    'thread2': dict(BASE, Ls=[16, 140, 220, 260, 264], tc=8.0, tb=0.0, a=0.05),
}


class Setup:
    def __init__(self, reg, fpow=1):
        if isinstance(reg, str) and '@' in reg:          # e.g. 'thread@0.5': all block sizes scaled (finite-L test)
            name, fac = reg.split('@')
            p = dict(REGIMES[name]); p['Ls'] = [int(round(x * float(fac))) for x in p['Ls']]
        else:
            p = REGIMES[reg] if isinstance(reg, str) else reg
        self.p = p
        self.Ls = np.array(p['Ls']); self.Eb = np.array(p['E'], float)
        self.L = int(self.Ls.sum())
        self.blk = np.repeat(np.arange(len(self.Ls)), self.Ls)
        self.En = self.Eb[self.blk]
        f2 = np.array(p['f2'], float) ** fpow          # fpow=2 : alternative reading F|B_i> with f already in B_i
        self.fn = np.sqrt(f2[self.blk])
        a = p['a']; Eb = self.Eb
        dE = Eb[:, None] - Eb[None, :]
        self.s0 = np.exp(-a * dE**2)
        self.s1 = np.exp(-a * dE**2 + 1.0 * Eb[:, None] - 0.5 * Eb[None, :])
        self.sV = np.exp(-0.25 * dE**2 + 0.3 * Eb[None, :])
        # K(al,be,ga) = exp(0.5 E_al - 0.6 E_be + 0.2 E_ga) is separable
        self.Ka = np.exp(0.5 * self.En); self.Kb = np.exp(-0.6 * self.En); self.Kc = np.exp(0.2 * self.En)
        self.A = np.exp(-p['tc'] * self.En / 2)               # e^{-t_c H/2}
        self.Bv = np.exp(-p['tb'] * self.En / 2) * self.fn     # e^{-t_b H/2} F

    def gauss(self, rng, prof):
        L = self.L
        sd = np.sqrt(prof[self.blk][:, self.blk] / L / 2)
        return sd * (rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L)))

    def draw_matter(self, rng):
        return dict(O=self.gauss(rng, self.s0), Ot=self.gauss(rng, self.s0),
                    Z=self.gauss(rng, self.s1), V=self.gauss(rng, self.sV))

    def W_of(self, V, O):
        return (self.Ka[:, None] * V * self.Kb[None, :]) @ O * self.Kc[None, :]

    def Ymat(self, V, O, y):
        tc = self.p['tc']
        t1 = tc / 4 + 1j * y; t2 = tc / 4 - 1j * y
        e1 = np.exp(-t1 * self.En); e2 = np.exp(-t2 * self.En)
        return (e1[:, None] * V * e2[None, :]) @ O * self.Bv[None, :]

    def Xmat(self, O):
        return self.A[:, None] * O * self.Bv[None, :]

    def brane(self, rng, k):
        """returns matrix T with C C^dag = T T^dag statistics; for k<L the actual C (L x k)."""
        L = self.L
        if k < L:
            return (rng.standard_normal((L, k)) + 1j * rng.standard_normal((L, k))) / np.sqrt(2)
        T = np.tril((rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))) / np.sqrt(2), -1)
        d = np.sqrt(rng.gamma(shape=k - np.arange(L), scale=1.0))
        T[np.diag_indices(L)] = d
        return T


def petz_quantities(Psi0, Psi1, Phis, want_eig=False):
    """Psi0, Psi1: L x r matrices (normalised states). Phis: list of L x r matrices.
    returns rho10 list, rho00, (eig data)"""
    tau = (Psi0.conj().T @ Psi0 + Psi1.conj().T @ Psi1) / 2
    lam, U = np.linalg.eigh(tau)
    lam = np.clip(lam, 1e-300, None)
    isq = lam ** -0.5
    P0 = (Psi0 @ U) * isq[None, :]            # Psi0 U lam^{-1/2}
    P1 = (Psi1 @ U) * isq[None, :]
    Kmat = (P0.conj().T @ P1) / 2             # lam^{-1/2} U^dag Psi0^dag Psi1 U lam^{-1/2} /2   (in U basis)
    K00 = (P0.conj().T @ P0) / 2
    Psi0U = Psi0 @ U
    # rho = Tr( Phi^dag Psi0 U K U^dag )  = sum_{ij} (U^dag Phi^dag Psi0 U)_{ij} K_{ji}
    out = []
    for Phi in Phis:
        M = (Phi @ U).conj().T @ Psi0U
        out.append(np.sum(M * Kmat.T))
    M00 = Psi0U.conj().T @ Psi0U
    rho00 = np.sum(M00 * K00.T)
    if want_eig:
        return out, rho00, (lam, U)
    return out, rho00
