"""
Step 3j (audit item H1): test of the absorption-placement rule (Step 3h: with the projector Pi_1 the signal's cubic
vertex sits on P's OWN clock line) in the random-matrix model of Step 3i.

Idea.  In the Step 3i RMT the level-1 clock operator is O1 = Z1 + eps W,  W = K.(V O0)  (explicit cubic cumulant), and the
physical state is Phi = V(y) Psi_0 with the SAME O0 as the code state Psi_0.  At O(eps) the cumulant can then pair V(Phi)
with Phi's own O0 (vertex on P's clock line: "own placement") or with the O0 of another replica (vertex on another
replica's clock line while P's clock line threads unabsorbed: "other placement").  Pi_1 (P's clock in level 1) kills
the second class.

Absorbed-sector construction.  Draw an independent copy Ot of O0 (same variance profile) and use it both in the signal
channel W = K.(V Ot) and in the physical state Phi = V(y) e^{-...} Ot ... F|B>.  Ot appears nowhere else, so at O(eps)
the cumulant can only pair V(Phi) with Phi's own Ot: every surviving contraction has the vertex on P's own clock line,
with the outgoing psi_1 line free to end on any replica's O1 (as in the Step 3h threading sum).  This realises the
Pi_1-projected observable.  Same random numbers are used for the unprojected (Step 3i) model, so
    true - absorbed  =  other-placement contribution  (sample by sample).

Prediction to test: absorbed-sector RMT rho_10 = Step 3h threading formula (shared energy = row block of V), up to the
cap residual seen in rho_00 and in the vertex-free coherence.
"""
import numpy as np, sys, json, time
sys.path.insert(0, '03_phase3_quantum/code')
from step3i_rmt import RMT, REGIMES

REGIMES = dict(REGIMES)
REGIMES['thread2deg'] = dict(REGIMES['thread2'], deg=True)     # degenerate code: level 1 has the level-0 profile
REGIMES['threaddeg'] = dict(REGIMES['thread'], deg=True)

class RMTAbs(RMT):
    def __init__(self, p):
        super().__init__(p)
        if p.get('deg'):
            self.s1 = self.s0.copy()
    def draw2(self, rng):
        O0, Z1, V = self._gauss(rng, self.s0), self._gauss(rng, self.s1), self._gauss(rng, self.sV)
        Ot = self._gauss(rng, self.s0)
        sl = self.slices
        def Wof(X):
            W = np.zeros((self.L, self.L), complex)
            for b in range(self.nb):
                VX = [V[sl[a], sl[b]] @ X[sl[b], :] for a in range(self.nb)]
                for a in range(self.nb):
                    W[sl[a], :] += VX[a]*self.K[a, b, self.blk][None, :]
            return W
        out = {}
        for nm, X in (('true', O0), ('abs', Ot)):
            W = Wof(X)
            A0 = self.Dc[:, None]*O0*self.DbF[None, :]
            A1Z = self.Dc[:, None]*Z1*self.DbF[None, :]; A1W = self.Dc[:, None]*W*self.DbF[None, :]
            AP = (self.D1[:, None]*V*self.D2[None, :]) @ (X*self.DbF[None, :])
            AQ = np.exp(-1j*self.p['y']*self.E)[:, None]*A1Z
            out[nm] = dict(A0=A0, A1Z=A1Z, A1W=A1W, AP=AP, AQ=AQ)
        return out

if __name__ == "__main__":
    nsamp = int(sys.argv[1]); regime = sys.argv[2]
    ks = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else [48, 300, 1600, 8000, 20000]
    P = REGIMES[regime]; M = RMTAbs(P); eps = P['eps']
    print(f"regime '{regime}': thread/cap = {M.thread_cap_ratio():.3g}, L = {M.L}, eps = {eps}", flush=True)
    acc = {k: {'true': [], 'abs': [], 'r00': []} for k in ks}; t0 = time.time()
    for smp in range(nsamp):
        rng = np.random.default_rng(7000 + smp); AA = M.draw2(rng)
        for k in ks:
            Q = M.wishart(np.random.default_rng(9000 + 97*smp + k), k)
            for nm in ('true', 'abs'):
                o = M.decode(AA[nm], Q, eps)
                acc[k][nm].append(dict(r10=(o[+1]['r10'] - o[-1]['r10'])/2/eps, phys=(o[+1]['phys'] - o[-1]['phys'])/2/eps))
                if nm == 'true': acc[k]['r00'].append(o['r00'])
        print(f"sample {smp} done ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    f = lambda z: f"{z.real:+.4f}{z.imag:+.4f}i"
    print(f"\n{'k':>6} {'x':>6} | {'RMT unprojected':>18} | {'RMT absorbed (Pi_1)':>20} {'+-':>7} | {'threading pred':>18} | {'other = unproj-abs':>18} | phys abs / planar | rho00")
    Sx = None
    for k in ks:
        a = acc[k]
        rt = np.array([v['r10'] for v in a['true']]); ra = np.array([v['r10'] for v in a['abs']]); d = rt - ra
        pa = np.mean([v['phys'] for v in a['abs']]); pt = np.mean([v['phys'] for v in a['true']])
        pe = M.planar(np.log(k)); thr = pe['r10']['V-row (threading)']/eps
        Ls = M.Ls; rho = 2*Ls; E = M.Eb; f2 = np.asarray(P['f2']); br = Ls*np.exp(-P['tb']*E)*f2
        y0 = Ls*np.exp(-P['tc']*E)*(M.s0 @ br); y1 = Ls*np.exp(-P['tc']*E)*(M.s1 @ br); pb = 0.5*(y0/y0.sum() + y1/y1.sum())
        x = np.log(k) + (pb*np.log(pb/rho)).sum()
        err_a = np.std(ra)/np.sqrt(len(ra)); err_d = np.std(d)/np.sqrt(len(d))
        r00 = np.mean(a['r00'])
        print(f"{k:6d} {x:+6.2f} | {f(rt.mean()):>18} | {f(ra.mean()):>20} {abs(err_a):7.1e} | {f(thr):>18} | {f(d.mean()):>18} (+-{abs(err_d):.1e}) | {f(pa)} / {f(pe['phys']/eps)} | {r00:.4f}/{pe['r00']:.4f}", flush=True)
        rows.append(dict(k=k, x=float(x), unproj=[rt.mean().real, rt.mean().imag], absorbed=[ra.mean().real, ra.mean().imag], absorbed_err=float(abs(err_a)),
                         threading=[thr.real, thr.imag], other=[d.mean().real, d.mean().imag], other_err=float(abs(err_d)),
                         phys_abs=[pa.real, pa.imag], phys_true=[pt.real, pt.imag], phys_planar=[(pe['phys']/eps).real, (pe['phys']/eps).imag],
                         rho00_rmt=float(r00), rho00_planar=float(pe['r00'] + (1 - pe['r00'])/2)))
    json.dump(dict(regime=regime, thread_cap_ratio=float(M.thread_cap_ratio()), nsamp=nsamp, rows=rows),
              open(f'03_phase3_quantum/results/step3j_absorption_{regime}.json', 'w'), indent=1)
