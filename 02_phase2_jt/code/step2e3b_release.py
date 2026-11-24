"""
Step 2E.3b -- an exactly solvable release protocol for the massive clock C1.

Geometric fact (exact, any JT vacuum): the zero set of a linear form is a geodesic.
Hence a dilaton kink can only sit on a geodesic, and a clock hovering on the (non-geodesic) station
orbit with *localised* support has NO kink: before release the exterior and the deep region are the
same vacuum Z_4. At release (event S) the clock's form must be created out of forms vanishing at S:
    mu P = nu N_a - nu' N_b,     N_a = (a_S^2, -a_S, 1) [null, on a = a_S],  N_b = (b_S^2, -b_S, 1),
i.e. C1 (+ m eps) is accompanied by an ingoing null shell along a = a_S and an outgoing null shell along b = b_S.
Regions (late exterior R3 = phi_evap, the background we already have):
    outside J+(S)            : phi_evap - nu' n_b      (= pre-release vacuum Z_4 = Z_1)
    J+(S), boundary side of C1 (R3): phi_evap
    J+(S), far side of C1 (R2)    : phi_evap + mu p
The outgoing shell reaches the boundary at y_out(S) and changes the boundary energy by E_3 - E_4.
"""
import numpy as np, sys, json
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
from step2d_clock import StaticJT, Drop, gd
from scipy.optimize import brentq

ip = lambda Z, W: (Z[0]*W[2] + Z[2]*W[0])/2 - Z[1]*W[1]

def form_val_grad(F, ah, bh):
    A, B, C = F; N = A + B*(ah + bh) + C*ah*bh; d = bh - ah
    return N/d, ((B + C*bh)*d + N)/d**2, ((B + C*ah)*d - N)/d**2

class Release:
    def __init__(self, M, D, mu):
        self.M, self.D, self.mu = M, D, mu
        self.P = np.array([D.t0**2 + D.R**2, -D.t0, 1.])/D.R
        self.aS, self.bS = -1.0, -D.rS
        self.Na = np.array([self.aS**2, -self.aS, 1.]); self.Nb = np.array([self.bS**2, -self.bS, 1.])
        x = np.linalg.lstsq(np.column_stack([self.Na, -self.Nb]), mu*self.P, rcond=None)[0]   # mu P = nu Na - nu' Nb (exact: P vanishes at S)
        self.nu, self.nup = x
        self.res = np.max(np.abs(self.nu*self.Na - self.nup*self.Nb - mu*self.P))
    def region(self, ah, bh):
        if ah >= self.aS and bh >= self.bS:
            p = form_val_grad(self.P, ah, bh)[0]; return 'R3' if p > 0 else 'R2'
        return 'old'
    def delta(self, ah, bh, reg):
        if reg == 'R3': return 0.0, 0.0, 0.0
        if reg == 'R2': v, ga, gb = form_val_grad(self.P, ah, bh); return self.mu*v, self.mu*ga, self.mu*gb
        v, ga, gb = form_val_grad(self.Nb, ah, bh); return -self.nup*v, -self.nup*ga, -self.nup*gb
    def grad(self, Y, bh, yB, reg):
        M = self.M; P = M.pieces(Y); ah = -np.exp(P['lG'] - self.D.lGS); r = bh/ah
        Nga = P['n0'] + (1 + r)*P['C1'] - r*P['m2']; Na_ = -P['C1'] + r*P['m2']; Nb_ = -P['C1'] + P['m2']
        dY = P['w']*(Na_*(1 - r) + Nga)/(1 - r)**2
        dbh = (Nb_*(1 - r) - Nga)/((-ah)*(1 - r)**2)
        fB = -np.exp(M.lG(yB) - self.D.lGS); k2 = 2*M.k
        gY = dY + k2*(-1/(yB - Y) - P['u'] + P['w']/(1 - r))
        gb = dbh + k2*(1/(bh - fB) - 1/(bh - ah))
        _, da, db = self.delta(ah, bh, reg)
        return np.array([gY + da*(-ah*P['w']), gb + db])
    def Sgen(self, Y, bh, yB, reg):
        M = self.M; P = M.pieces(Y); ah = -np.exp(P['lG'] - self.D.lGS)
        phi = M.phi_r(Y, bh/ah, P) + self.delta(ah, bh, reg)[0]
        fB = -np.exp(M.lG(yB) - self.D.lGS)
        return phi + 2*M.k*(np.log(yB - Y) + np.log(abs(bh - fB)) - P['lpsi'] - np.log(bh - ah))
    def find(self, yB, reg, Ys, bgrid):
        """bracketed search: for each Y the roots of d_b S on bgrid, then sign changes of d_Y S along each branch."""
        out = []
        branches = []
        for Y in Ys:
            g = np.array([self.grad(Y, b, yB, reg)[1] for b in bgrid])
            idx = np.where(np.isfinite(g[:-1]) & np.isfinite(g[1:]) & (np.sign(g[:-1]) != np.sign(g[1:])))[0]
            roots = []
            for i in idx:
                b = brentq(lambda bb: self.grad(Y, bb, yB, reg)[1], bgrid[i], bgrid[i+1], xtol=1e-18)
                if abs(self.grad(Y, b, yB, reg)[1]) < 1e-6:
                    ah = -np.exp(self.M.lG(Y) - self.D.lGS)
                    if self.region(ah, b) == reg: roots.append(b)
            branches.append(roots)
        # follow the largest-b branch (the one near the horizon)
        vals = [(Y, max(rs)) for Y, rs in zip(Ys, branches) if rs]
        for (Y1, b1), (Y2, b2) in zip(vals[:-1], vals[1:]):
            g1 = self.grad(Y1, b1, yB, reg)[0]; g2 = self.grad(Y2, b2, yB, reg)[0]
            if np.sign(g1) != np.sign(g2) and abs(Y2 - Y1) < 1.5*(Ys[1] - Ys[0]):
                def h(Y):
                    gg = np.array([self.grad(Y, b, yB, reg)[1] for b in bgrid])
                    ii = np.where(np.sign(gg[:-1]) != np.sign(gg[1:]))[0]
                    bs = [brentq(lambda bb: self.grad(Y, bb, yB, reg)[1], bgrid[i], bgrid[i+1], xtol=1e-18) for i in ii]
                    bs = [b for b in bs if abs(self.grad(Y, b, yB, reg)[1]) < 1e-6]
                    return self.grad(Y, max(bs), yB, reg)[0], max(bs)
                try:
                    Yq = brentq(lambda Y: h(Y)[0], Y1, Y2, xtol=1e-11); bq = h(Yq)[1]
                    ah = -np.exp(self.M.lG(Yq) - self.D.lGS)
                    out.append(dict(Y=Yq, bh=bq, ah=ah, reg=self.region(ah, bq), S=self.Sgen(Yq, bq, yB, reg)))
                except Exception:
                    pass
        return out
    def tau_C1_at_b(self, bh):
        return gd(np.log((bh - self.D.t0)/self.D.R)) - self.D.tauH

def lam_pair(Z3, Z4):
    ph3, ph4 = np.sqrt(-ip(Z3, Z3)), np.sqrt(-ip(Z4, Z4)); return (ph3 - ph4)/(ph3 + ph4)

def static_check():
    """Energies of the release in the exact static black hole (hat coordinates, kappa = 1)."""
    S = StaticJT(); rows = []
    for ts in (np.pi/16, np.pi/8, np.pi/4, 3*np.pi/8):
        D = Drop(S, 3.0, 2/np.sin(ts)); G = np.exp(D.lGS)
        Z3 = np.array([0., -2., -2*G])
        for mu in (1e-3, 1e-2):
            R = Release(S, D, mu); eps = ip(Z3, R.P)
            Z2 = Z3 + mu*R.P; Z4 = Z3 - R.nup*R.Nb; E = lambda Z: -ip(Z, Z)/2
            rows.append(dict(ts=ts, mu=mu, decomposition_residual=float(R.res), nu=float(R.nu), nup=float(R.nup),
                             eps_minus_dtau_dy=float(eps - 2/np.tan(ts)),
                             E3_minus_E4_over_mu_eps=float((E(Z3) - E(Z4))/(mu*eps)),
                             E3_minus_E2_minus_mu_eps=float(E(Z3) - E(Z2) - mu*eps - mu**2/2),
                             lambda_34=float(lam_pair(Z3, Z4)), lambda_naive=float(mu*eps/(8*E(Z3)))))
    return rows

if __name__ == "__main__":
    rows = static_check()
    for r in rows: print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(rows, open('02_phase2_jt/results/step2e3b_static_release.json', 'w'), indent=1)


class ConsistentRelease(Release):
    """Self-consistent version: the station sits at fixed phi of the PRE-release dilaton phi_old = phi_evap - nu' n_b,
    C1 is released at rest relative to that station; phi_evap represents the POST-release exterior R3 (late times).
    Iterates nu' <-> (S, tangent, geodesic) to convergence."""
    def __init__(self, M, D0, mu, phi_s, n_iter=30):
        self.M, self.D, self.mu = M, D0, mu       # D0 supplies Y_S, lGS (hat normalisation) and helpers
        YS = D0.YS; nup = 0.0; self.hist = []
        for it in range(n_iter):
            # station radius at Y_S in phi_old (ah_S = -1 by normalisation)
            def phi_old(r):
                bh = -r; ah = -1.0
                v = M.phi_r(YS, r)
                if nup != 0.0:
                    Nb = np.array([bh**2, -bh, 1.]); v -= nup*form_val_grad(Nb, ah, bh)[0]
                return v - phi_s
            # NB: n_b vanishes at S itself, so phi_old(S) = phi_evap(S): r_S unchanged; the TANGENT changes
            rS = brentq(phi_old, 1e-9, 0.999999)
            aS, bS = -1.0, -rS
            Nb = np.array([bS**2, -bS, 1.]); Na = np.array([aS**2, -aS, 1.])
            # gradient of phi_old at S in hat coords
            P_ = M.pieces(YS); w = P_['w']; r = rS
            Nga = P_['n0'] + (1 + r)*P_['C1'] - r*P_['m2']; Na_ = -P_['C1'] + r*P_['m2']; Nb_ = -P_['C1'] + P_['m2']
            dY = w*(Na_*(1 - r) + Nga)/(1 - r)**2              # d_Y phi_evap
            dah = dY/(-aS*w)                                    # d_ah = d_Y/(d ah/dY), d ah/dY = -ah w
            dbh = (Nb_*(1 - r) - Nga)/((-aS)*(1 - r)**2)
            _, ga, gb = form_val_grad(Nb, aS, bS)
            dah -= nup*ga; dbh -= nup*gb
            ratio = -dah/dbh                                    # d bh / d ah along the level curve
            lamS = 0.5*np.log(ratio); R = (bS - aS)/(2*np.cosh(lamS)); t0 = aS + R*np.exp(-lamS)
            P = np.array([t0**2 + R**2, -t0, 1.])/R
            x = np.linalg.lstsq(np.column_stack([Na, -Nb]), mu*P, rcond=None)[0]
            nup_new = x[1]; self.hist.append(nup_new)
            if abs(nup_new - nup) < 1e-15: nup = nup_new; break
            nup = nup_new
        self.nu, self.nup = x; self.P = P; self.Na, self.Nb = Na, Nb; self.aS, self.bS = aS, bS
        self.t0, self.R, self.lamS = t0, R, lamS
        self.lamH = np.log(-t0/R); self.tauH = gd(self.lamH)
        self.res = np.max(np.abs(self.nu*Na - self.nup*Nb - mu*P))
        # geodesic tangent rotation relative to the test-clock drop
        self.dlamS = lamS - D0.lamS
    def tau_C1_at_b(self, bh):
        return gd(np.log((bh - self.t0)/self.R)) - self.tauH
    def tau_C1_at_a(self, ah):
        return gd(-np.log((self.t0 - ah)/self.R)) - self.tauH
