"""
Step 3i (open point 2) -- independent check of the replica threading in a finite random-matrix model
(PSSY appendix D, extended with matter as Gaussian ETH matrices and an explicit cubic cumulant).

Model
  B: energy blocks alpha = 0..nb-1 with L_alpha states each (energy E_alpha); R: dimension k.
  brane states  |B_i> = sum_n C_{ni} f_n |n>,  C i.i.d. complex Gaussian (E|C|^2 = 1).
  matter        O0, Z1, V: complex Gaussian, E|M_nm|^2 = s_M(alpha_n, alpha_m)/L  (block variance profiles)
                O1 = Z1 + eps W,  W_ln = sum_m K(alpha_l, alpha_m, alpha_n) V_lm O0_mn
                -> cubic cumulant  E[(O1*)_ln V_lm O0_mn] = eps K s_V s_0 / L^2  =: c(l, m, n)   (only coupling of O1 to V, O0)
  code states   Psi_a = sum_i  e^{-tc H/2} O_a e^{-tb H/2} F |B_i> |i>_R          (a = 0, 1; normalised)
  physical      Phi   = V(y) Psi_0 = sum_i e^{-tau1 H} V e^{-tau2 H} O0 e^{-tb H/2} F |B_i> |i>,  tau1,2 = tc/4 +- i y
  Petz          O_R = sigma^{-1/2} Tr_B(|Psi_1><Psi_0|) sigma^{-1/2},  sigma = (rho_0 + rho_1)/2
  decoded       rho_10 = <Phi| O_R |Psi_0>  (its O(eps) part, extracted as the odd part in eps with common random numbers),
                rho_00 = <Psi_0| (|Psi_0><Psi_0|)_R |Psi_0>  (= c1 of level 0)
  All R-side quantities are evaluated from L x L Gram matrices  X_a^dag X_b = conj(A_a) Q A_b^T,  Q = conj(C) C^T
  (complex Wishart, sampled by the Bartlett decomposition for k >= L), and sigma^{-1/2} = A G^{-3/2} A^dag with
  A = [X_0, X_1]/sqrt 2, G = A^dag A (pseudo-inverse on the support).

Planar prediction (Step 3h threading): shared energy = block of the row index of V (P's arc [u_+, y]); P's factor
  T(alpha) = L_alpha e^{-(tc/2 + tau1*) E_alpha} sum_{beta,gamma} L_beta L_gamma e^{-tau2* E_beta - tb E_gamma} f_gamma c(alpha,beta,gamma)
  rho_10 = sum_alpha T p0 J1 J0 / (sqrt(Z0 Z1) rho_alpha),   rho_00 = sum p0^2 J0^2/rho,   rho_alpha = d L_alpha,
with J_a exact (step3i_resolvent) or canonical.  Strawmen: the same formula with the shared energy put on the column
index of V (cap [y, u_-]) or on the brane-side index.
"""
import numpy as np, sys, json, time
sys.path.insert(0, '03_phase3_quantum/code')
from step3i_resolvent import ExactResolvent, lse

# ------------------------------------------------------------------ parameters
BASE = dict(Ls=[30, 120, 200, 250, 300], E=[0.0, 0.7, 1.4, 2.1, 2.8], y=0.9, eps=0.05, f2=[1.0, 0.8, 0.65, 0.5, 0.4])
# regimes: the central side (clock middle arc, length tc) cold and the brane side hot -> few effective central states ->
# threading pairings dominate ('thread'); the reverse ('cap'); and an intermediate case ('mixed').
REGIMES = {'thread': dict(BASE, tc=8.0, tb=0.05), 'cap': dict(BASE, tc=0.05, tb=8.0), 'mixed': dict(BASE, tc=1.2, tb=0.8),
           'thread2': dict(BASE, Ls=[16, 140, 220, 260, 264], tc=8.0, tb=0.0, a=0.05)}
P = REGIMES['mixed']
def profiles(E, a=0.4):
    E = np.asarray(E); dE = E[:, None] - E[None, :]
    s0 = np.exp(-a*dE**2)
    s1 = np.exp(-a*dE**2 + 1.0*E[:, None] - 0.5*E[None, :])       # level 1: very different central-energy weighting
    sV = np.exp(-0.25*dE**2 + 0.3*E[None, :])
    K = np.exp(0.5*E[:, None, None] - 0.6*E[None, :, None] + 0.2*E[None, None, :])
    return s0, s1, sV, K

class RMT:
    def thread_cap_ratio(self):
        """two-replica connected weights: threading (shared central face) / caps (shared brane face), level 0."""
        p = self.p; Ls, E, L = self.Ls, self.Eb, self.L; f2 = np.asarray(p['f2'])
        a2 = np.exp(-p['tc']*E); b2 = np.exp(-p['tb']*E)*f2
        thread = np.sum(Ls*a2**2*((self.s0/L) @ (Ls*b2))**2)
        cap = np.sum(Ls*b2**2*((Ls*a2) @ (self.s0/L))**2)
        return thread/cap
    def __init__(self, p=P):
        self.p = p; Ls = np.array(p['Ls']); self.Ls = Ls; self.nb = len(Ls); self.L = L = Ls.sum()
        self.blk = np.repeat(np.arange(self.nb), Ls); self.Eb = np.asarray(p['E']); E = self.Eb[self.blk]
        self.s0, self.s1, self.sV, self.K = profiles(p['E'], p.get('a', 0.4))
        tc, tb, y = p['tc'], p['tb'], p['y']; self.tau1, self.tau2 = tc/4 + 1j*y, tc/4 - 1j*y
        self.Dc = np.exp(-tc*E/2); self.DbF = np.exp(-tb*E/2)*np.sqrt(np.asarray(p['f2'])[self.blk])
        self.D1, self.D2 = np.exp(-self.tau1*E), np.exp(-self.tau2*E); self.E = E
        self.slices = [slice(int(a), int(b)) for a, b in zip(np.r_[0, np.cumsum(Ls)[:-1]], np.cumsum(Ls))]

    def _gauss(self, rng, s):
        L = self.L; S = np.sqrt(s[self.blk][:, self.blk]/L)
        return S*(rng.standard_normal((L, L)) + 1j*rng.standard_normal((L, L)))/np.sqrt(2)

    def draw(self, rng):
        O0, Z1, V = self._gauss(rng, self.s0), self._gauss(rng, self.s1), self._gauss(rng, self.sV)
        W = np.zeros((self.L, self.L), complex); sl = self.slices
        for b in range(self.nb):
            VO = [V[sl[a], sl[b]] @ O0[sl[b], :] for a in range(self.nb)]          # (rows a) x (all cols)
            for a in range(self.nb):
                Kc = self.K[a, b, self.blk]                                          # K(a, b, gamma_n) per column n
                W[sl[a], :] += VO[a]*Kc[None, :]
        A0 = self.Dc[:, None]*O0*self.DbF[None, :]
        A1Z = self.Dc[:, None]*Z1*self.DbF[None, :]; A1W = self.Dc[:, None]*W*self.DbF[None, :]
        AP = (self.D1[:, None]*V*self.D2[None, :]) @ (O0*self.DbF[None, :])
        AQ = np.exp(-1j*self.p['y']*self.E)[:, None]*A1Z                          # Phi' = e^{-i y H} Psi_1 (vertex-free)
        return dict(A0=A0, A1Z=A1Z, A1W=A1W, AP=AP, AQ=AQ)

    def wishart(self, rng, k):
        L = self.L
        if k < L:
            C = (rng.standard_normal((L, k)) + 1j*rng.standard_normal((L, k)))/np.sqrt(2)
            return np.conj(C) @ C.T
        T = np.tril((rng.standard_normal((L, L)) + 1j*rng.standard_normal((L, L)))/np.sqrt(2), -1)
        T[np.diag_indices(L)] = np.sqrt(rng.gamma(k - np.arange(L), 1.0))
        return T @ T.conj().T

    def decode(self, A, Q, eps):
        """returns rho_10(+eps), rho_10(-eps), rho_00, physical <Psi1|Phi>(+-eps) -- all for normalised states."""
        B = {nm: Q @ M.T for nm, M in A.items()}
        G = lambda a, b: np.conj(A[a]) @ B[b]
        g00, g0Z, g0W, gZZ, gZW, gWW = G('A0', 'A0'), G('A0', 'A1Z'), G('A0', 'A1W'), G('A1Z', 'A1Z'), G('A1Z', 'A1W'), G('A1W', 'A1W')
        gP0, gPZ, gPW = G('AP', 'A0'), G('AP', 'A1Z'), G('AP', 'A1W'); gQ0, gQZ = G('AQ', 'A0'), G('AQ', 'A1Z')
        n0, n1 = np.trace(g00).real, np.trace(gZZ).real
        out = {}
        for sgn in (+1, -1):
            e = sgn*eps
            G00 = g00/n0; G01 = (g0Z + e*g0W)/np.sqrt(n0*n1); G11 = (gZZ + e*(gZW + gZW.conj().T) + e**2*gWW)/n1
            GP0 = gP0/n0; GP1 = (gPZ + e*gPW)/np.sqrt(n0*n1)
            Gm = 0.5*np.block([[G00, G01], [G01.conj().T, G11]])
            ev, U = np.linalg.eigh(Gm); keep = ev > 1e-11*ev.max()
            Gm32 = (U[:, keep]*ev[keep]**-1.5) @ U[:, keep].conj().T
            AX0 = np.vstack([G00, G01.conj().T])/np.sqrt(2)                     # A^dag X_0
            AX1 = np.vstack([G01, G11])/np.sqrt(2)                              # A^dag X_1
            AXP = np.vstack([GP0.conj().T, GP1.conj().T])/np.sqrt(2)            # A^dag X_Phi
            M1 = Gm32 @ (AX1 @ AX0.conj().T); M2 = Gm32 @ (AX0 @ AXP.conj().T)
            r10 = np.sum(M1*M2.T)/2                                             # Tr(M1 M2)/d  (Petz normalised: 1_code -> 1)
            GQ0 = gQ0/np.sqrt(n0*n1); AXQ = np.vstack([GQ0.conj().T, (gQZ/n1).conj().T])/np.sqrt(2)
            r10q = np.sum(M1*(Gm32 @ (AX0 @ AXQ.conj().T)).T)/2
            out[sgn] = dict(r10=r10, phys=np.trace(GP1), rank=int(keep.sum()), r10q=r10q, physq=np.trace(gQZ/n1))   # <Phi|Psi_1>
            if sgn == +1:
                M0 = Gm32 @ (AX0 @ AX0.conj().T); out['r00'] = np.sum(M0*M0.T).real/2
        return out

    # ---------------------------------------------------------------- planar prediction
    def planar(self, logk, which='exact'):
        p = self.p; Ls, E, L = self.Ls, self.Eb, self.L; f2 = np.asarray(p['f2']); tc, tb = p['tc'], p['tb']; eps = p['eps']
        br = Ls*np.exp(-tb*E)*f2                                                  # brane-side block weights
        y0 = Ls*np.exp(-tc*E)*(self.s0 @ br)/L; y1 = Ls*np.exp(-tc*E)*(self.s1 @ br)/L
        Z0, Z1 = y0.sum(), y1.sum(); p0, p1 = y0/Z0, y1/Z1; rho = 2*Ls
        c = eps*self.K*self.sV[:, :, None]*self.s0[None, :, :]/L**2                # c(alpha, beta, gamma)
        summ = (Ls*np.exp(-(tc/2 + np.conj(self.tau1))*E))[:, None, None]*(Ls*np.exp(-np.conj(self.tau2)*E))[None, :, None] \
               *br[None, None, :]*c                                                # T integrand (alpha=V row, beta=V col, gamma=brane)
        T = {'V-row (threading)': summ.sum(axis=(1, 2)), 'V-col (strawman)': summ.sum(axis=(0, 2)), 'brane (strawman)': summ.sum(axis=(0, 1))}
        phys = summ.sum()/np.sqrt(Z0*Z1)
        if np.isfinite(logk) and which == 'exact':
            R = ExactResolvent(np.log(rho), [np.log(p0), np.log(p1)], logk, M=4096); lJ0, lJ1 = R.logJ(); J0, J1 = np.exp(lJ0), np.exp(lJ1)
            cnt = R.sum_rules()[0]
        elif np.isfinite(logk):
            R = ExactResolvent(np.log(rho), [np.log(p0), np.log(p1)], logk, M=256); lJ0, lJ1 = R.logJ_canonical(); J0, J1 = np.exp(lJ0), np.exp(lJ1); cnt = None
        else:
            J0, J1 = (p0/rho)**-0.5, (p1/rho)**-0.5; cnt = None
        r00 = np.sum(p0**2/rho*J0**2)
        y = p['y']; Tq = Ls*np.exp(-(tc - 1j*y)*E)*(self.s1 @ br)/L                  # P-factor of Phi' at shared central energy
        # P = <Phi'|.|Psi_0>: T_q/sqrt(Z0 Z1);  D = <Psi_0|.|Psi_1>: p0 sqrt(Z0/Z1)  ->  product T_q p0 / Z1
        self.last_q = dict(r10q=np.sum(Tq*p0*J1*J0/rho)/Z1, physq=Tq.sum()/Z1,
                           r10q_brane=None)
        r10 = {nm: np.sum(t*p0*J1*J0/rho)/np.sqrt(Z0*Z1) for nm, t in T.items()}
        return dict(r10=r10, r00=r00, phys=phys, count=cnt)

if __name__ == "__main__":
    regime = sys.argv[2] if len(sys.argv) > 2 else 'mixed'; P = REGIMES[regime]
    M = RMT(P); L = M.L; nsamp = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"regime '{regime}': tc = {P['tc']}, tb = {P['tb']};  two-replica threading/cap weight ratio = {M.thread_cap_ratio():.3g}", flush=True)
    ks = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else [48, 120, 300, 700, 1600, 3600, 8000, 20000]
    Sbh = None
    res = dict(regime=regime, thread_cap_ratio=float(M.thread_cap_ratio()), params={k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in P.items()}, L=int(L), rows=[])
    acc = {k: [] for k in ks}; t0 = time.time()
    for smp in range(nsamp):
        rng = np.random.default_rng(1000 + smp); A = M.draw(rng)
        for k in ks:
            Q = M.wishart(np.random.default_rng(5000 + 97*smp + k), k)
            o = M.decode(A, Q, P['eps'])
            r10 = (o[+1]['r10'] - o[-1]['r10'])/2; ph = (o[+1]['phys'] - o[-1]['phys'])/2
            acc[k].append(dict(r10=r10, r00=o['r00'], phys=ph, rank=o[+1]['rank'], r10q=(o[+1]['r10q'] + o[-1]['r10q'])/2, physq=o[+1]['physq']))
        print(f"sample {smp} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nL = {L}, d e^S = 2L = {2*L};  eps = {P['eps']}  (rho_10 is O(eps); values below are divided by eps)")
    hdr = f"{'k':>6} {'x=log k-S':>9} | {'RMT rho10/eps':>24} {'+-':>6} | {'planar exact (threading)':>24} | {'canonical':>20} | {'V-col strawman':>20} | {'brane strawman':>20} | rho00 RMT / planar"
    print(hdr)
    pl_inf = M.planar(np.inf)
    f = lambda z: f"{z.real:+.4e}{z.imag:+.4e}i"
    for k in ks:
        a = acc[k]; r = np.array([v['r10'] for v in a])/P['eps']; r00 = np.array([v['r00'] for v in a]); ph = np.array([v['phys'] for v in a])/P['eps']
        pe = M.planar(np.log(k)); pc = M.planar(np.log(k), 'canonical')
        # x relative to the code-averaged entropy
        if Sbh is None:
            Ls = M.Ls; rho = 2*Ls; E = M.Eb; f2 = np.asarray(P['f2']); br = Ls*np.exp(-P['tb']*E)*f2
            y0 = Ls*np.exp(-P['tc']*E)*(M.s0 @ br); y1 = Ls*np.exp(-P['tc']*E)*(M.s1 @ br); pb = 0.5*(y0/y0.sum() + y1/y1.sum())
            Sbh = -(pb*np.log(pb/rho)).sum()
        mr, sr = r.mean(), r.std()/np.sqrt(len(r))
        thr = pe['r10']['V-row (threading)']/P['eps']
        row = dict(k=k, x=float(np.log(k) - Sbh), rmt=[mr.real, mr.imag], rmt_err=float(abs(sr)), rmt_phys=[ph.mean().real, ph.mean().imag],
                   planar_exact=[thr.real, thr.imag], planar_canonical=[(pc['r10']['V-row (threading)']/P['eps']).real, (pc['r10']['V-row (threading)']/P['eps']).imag],
                   strawman_Vcol=[(pe['r10']['V-col (strawman)']/P['eps']).real, (pe['r10']['V-col (strawman)']/P['eps']).imag],
                   strawman_brane=[(pe['r10']['brane (strawman)']/P['eps']).real, (pe['r10']['brane (strawman)']/P['eps']).imag],
                   rho00_rmt=float(r00.mean()), rho00_planar=float(pe['r00']), rho00_canonical=float(pc['r00']), count=pe['count'], phys_planar=[(pe['phys']/P['eps']).real, (pe['phys']/P['eps']).imag])
        rq = np.array([v['r10q'] for v in a]); M.planar(np.log(k)); q = M.last_q
        row.update(rmt_q=[rq.mean().real, rq.mean().imag], rmt_q_err=float(abs(rq.std())/np.sqrt(len(rq))), planar_q=[q['r10q'].real, q['r10q'].imag],
                   phys_q_rmt=[np.mean([v['physq'] for v in a]).real, np.mean([v['physq'] for v in a]).imag], phys_q_planar=[q['physq'].real, q['physq'].imag])
        print(f"        vertex-free Phi' = e^(-iyH)Psi_1:  RMT rho10 {f(rq.mean())} (+-{abs(rq.std())/np.sqrt(len(rq)):.1e})   planar threading {f(q['r10q'])}   [physical <Phi'|Psi_1>: RMT {f(np.mean([v['physq'] for v in a]))} planar {f(q['physq'])}]", flush=True)
        res['rows'].append(row)
        print(f"{k:6d} {row['x']:+9.2f} | {f(mr):>24} {abs(sr)/abs(mr):6.1e} | {f(thr):>24} | {f(pc['r10']['V-row (threading)']/P['eps']):>20} | "
              f"{f(pe['r10']['V-col (strawman)']/P['eps']):>20} | {f(pe['r10']['brane (strawman)']/P['eps']):>20} | {r00.mean():.4f} / {pe['r00']:.4f} (canon {pc['r00']:.4f})", flush=True)
    print(f"\none-sided <Phi|Psi1>/eps: RMT {f(np.mean([v['phys'] for v in acc[ks[-1]]])/P['eps'])}  planar {f(pl_inf['phys']/P['eps'])}")
    print(f"deep post-Page planar rho10/eps (threading) = {f(pl_inf['r10']['V-row (threading)']/P['eps'])}")
    res['deep_planar'] = {nm: [(v/P['eps']).real, (v/P['eps']).imag] for nm, v in pl_inf['r10'].items()}
    json.dump(res, open(f'03_phase3_quantum/results/step3i_rmt_{regime}.json', 'w'), indent=1, default=float)
