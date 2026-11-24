"""Step 2D runs: evaporating background vs static formulas at the instantaneous kappa."""
import numpy as np, sys, json, time
sys.path.insert(0, '02_phase2_jt/code')
from step2_logjt import LogJT
from step2d_clock import Drop, static_loop, gd
from scipy.optimize import brentq

def run(M, ydrop_target, ts_target, do_qes=False):
    kap0 = M.kappa(ydrop_target); phi_s = 2*kap0/np.sin(ts_target)
    # choose Y_S so that the radar time of S equals ydrop_target (iterate: radar = (YS + Yout)/2)
    YS = ydrop_target - 0.5
    for _ in range(40):
        D = Drop(M, YS, phi_s); YS += (ydrop_target - D.y_radar)
    D = Drop(M, YS, phi_s); yr = D.y_radar; kap = M.kappa(yr)
    ts_inst = np.arcsin(2*kap/phi_s)
    le = D.lam_end(0.0); tau_end = gd(le)
    aend = D.point(le)[0]; Yend = D.Y_of_ah(aend)
    # far end of C1 inside the Poincare patch: b -> infinity (inner/Cauchy horizon of the throat), a -> t0
    tau_inner = np.pi/2 - D.tauH                     # proper time from EH crossing to b = infinity
    Y_inner = D.Y_of_ah(D.t0); phi_inner = D.phi_at(30.0 + D.lamH)
    res = dict(k=M.k, y_radar=yr, kappa=kap, phi_s=phi_s, ts_inst=ts_inst, YS=YS, Y_H=D.Y_of_ah(D.point(D.lamH)[0]),
               drop_to_phi0=tau_end - D.tauS, d1=tau_end - D.tauS - np.pi/2,
               hor_to_phi0=tau_end - D.tauH, d2=(tau_end - D.tauH) - ts_inst,
               Yend=Yend, d3_rindler=2*kap*(Yend - yr),
               tau_inner=tau_inner, tau_inner_static=2*ts_inst, phi_inner=phi_inner, phi_h_inst=2*kap,
               Y_inner_minus_radar_rindler=2*kap*(Y_inner - yr), static_inner_rindler=np.log(1/np.cos(ts_inst)))
    # loop: compare at fixed station proper time T2 (invariant)
    T2max = None
    # T2 at which the meeting point reaches phi = 0 on C1:  aE = aend
    def g(T):
        YE = D.YE_of_T2(T); return -np.exp(M.lG(YE) - D.lGS) - aend
    Ts_static_s = np.log((1 + np.sin(ts_inst))/np.cos(ts_inst))/np.tan(ts_inst)
    T2max = brentq(g, 1e-3, 3*Ts_static_s)
    Ts = np.linspace(0.05, 1.0, 8)*T2max
    L = [D.loop(T)[0] for T in Ts]; Lst = [static_loop(T, ts_inst) for T in Ts]
    res.update(T2_at_phi0=T2max, T2_at_phi0_static=Ts_static_s,
               loop_evap=list(map(float, L)), loop_static=list(map(float, Lst)), T2=list(map(float, Ts)),
               loop_dev_max=float(np.max(np.abs(np.array(L) - np.array(Lst)))),
               loop_at_phi0=float(L[-1]), loop_at_phi0_static=float(static_loop(Ts_static_s, ts_inst)))
    if do_qes:
        # read-out time: earliest yB with a_Q(yB) >= a_end  <=>  yB - Delta(yB) = Yend
        def h(yB):
            Y, eta = M.qes(yB); return Y - Yend
        Dg = np.log(4*kap/M.k)/(2*kap)
        yread = brentq(h, Yend + 0.5*Dg, Yend + 3*Dg, xtol=1e-9)
        Yq, eta = M.qes(yread); Delta = yread - Yq
        yread_in = brentq(lambda yB: M.qes(yB)[0] - Y_inner, Y_inner + 0.5*Dg, Y_inner + 3*Dg, xtol=1e-9)
        Delta_in = yread_in - M.qes(yread_in)[0]
        # sliver near the horizon not yet in D(I) at yread: tau between EH crossing and b = b_Q(yread)
        bQ = eta*np.exp(M.lG(yread) - D.lGS)
        lam_b = np.log((bQ - D.t0)/D.R); sliver = gd(lam_b) - D.tauH
        # crossing of the QES locus (Gyongyosi-type "radiated-out" point) for the test clock
        def F(yB):
            Y, e = M.qes(yB); ahQ = -np.exp(M.lG(Y) - D.lGS); bhQ = e*np.exp(M.lG(yB) - D.lGS)
            bC1 = D.t0 + D.R**2/(D.t0 - ahQ); return bC1 - bhQ
        grid = res['Y_H'] + np.linspace(0.2*Dg, 4*Dg, 25); Fv = [F(y) for y in grid]
        j = [i for i in range(len(grid)-1) if np.sign(Fv[i]) != np.sign(Fv[i+1])]
        yx = brentq(F, grid[j[0]], grid[j[0]+1], xtol=1e-9)
        Y, e = M.qes(yx); ahQ = -np.exp(M.lG(Y) - D.lGS); tau_x = gd(D.lam_of_ah(ahQ)) - D.tauH
        res.update(y_read_inner=yread_in, read_inner_minus_static=yread_in - (yr + np.log(1/np.cos(ts_inst))/(2*kap) + Delta_in),
                   y_read=yread, Delta_at_read=Delta, read_delay_after_radar=yread - yr,
                   static_read=yr + Delta, read_minus_static=yread - (yr + Delta),
                   sliver_at_read=sliver, sliver_frac=sliver/res['hor_to_phi0'],
                   y_cross_QES=yx, tau_cross_QES=tau_x, endurance_frac=tau_x/res['hor_to_phi0'])
    return res

if __name__ == "__main__":
    rows = []
    for k in (0.05, 0.02, 0.01, 3e-3, 1e-3, 3e-4):
        M = LogJT(k)
        for x in (0.5, 1.5):
            for ts in (np.pi/8, np.pi/4, 3*np.pi/8):
                t = time.time()
                r = run(M, x/k, ts, do_qes=(abs(ts - np.pi/4) < 1e-9))
                rows.append(r)
                s = (f"k={k:<6g} x={x} ts={ts:.3f} kap={r['kappa']:.3f} | d1={r['d1']:+.2e} d2={r['d2']:+.2e} "
                     f"2kap(Yend-yr)={r['d3_rindler']:+.2e} | loop dev={r['loop_dev_max']:.2e} "
                     f"T2*(evap/static)={r['T2_at_phi0']/r['T2_at_phi0_static']:.5f} | inner: tau {r['tau_inner']:.5f} vs 2ts {r['tau_inner_static']:.5f}, phi {r['phi_inner']:.4f} vs -phi_h {-r['phi_h_inst']:.4f}")
                if 'y_read' in r:
                    s += (f"\n      read-out: y_read-y_radar={r['read_delay_after_radar']:.4f} (Delta={r['Delta_at_read']:.4f}), "
                          f"minus static={r['read_minus_static']:+.2e}; inner-horizon read minus static={r['read_inner_minus_static']:+.2e}; sliver/tau_int={r['sliver_frac']:.2e}; "
                          f"QES-locus crossing tau_x/tau_int={r['endurance_frac']:.3e}")
                print(s + f"  [{time.time()-t:.0f}s]", flush=True)
    json.dump(rows, open('02_phase2_jt/results/step2d_runs.json', 'w'), indent=1, default=float)
