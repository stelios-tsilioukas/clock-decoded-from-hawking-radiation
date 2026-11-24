"""Step 3h checks: (1) Sum_s T(s) = exact one-sided amplitudes of Steps 3b and 3e;  (2) deep post-Page decoding
reproduces them when the two levels have equal weights;  (3) structure of the omega != 0 difference."""
import numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '03_phase3_quantum/code')
import step3h_replica as H, step3e_matter_clock as M, step3d_backreaction as B
ts = np.pi/4
for C in (10.0, 2.0):
    X = H.Replica(C, ts, 2.0); s = X.s
    # (1a) tick-rate moments (3b): weights (p/2, p/2, m - p/2)
    ref = B.Exact(C)
    for dy in (-0.347, 0.0):
        ly = X.log_y(2.0); m3b = ref.moments(ts, np.array([dy]), 2.0, (1, 2))
        for p in (1, 2):
            lT = X.log_T(dy, p/2, p/2, 2.0 - p/2)
            val = np.exp(H.lse(lT) - H.lse(ly))*s**p
            print(f"C={C} dy={dy:+.3f} p={p}: Sum T / Sum y * s^p = {val.real:.10f}{val.imag:+.1e}i   Step 3b: {m3b[p][0].real:.10f}")
    # (1b) 3e amplitude, unequal weights: <O_D(u-) V O_{D+w}(u+)>  vs  Exact.log_triangle with the same chords
    D, om, h = 2.0, 0.4, 1.0; wa, wb, wc = (h + om)/2, (h - om)/2, D + (om - h)/2
    E = M.Exact(C, N=2400)
    for dy in (-0.347, 0.0):
        lT = X.log_T(dy, wa, wb, wc); lref = E.log_triangle(ts, [dy], wa, wb, wc, m_grid=D + om/2)[0]
        # log_triangle = log(fan/Z) + 2(wa+wb+wc) log s ; compare ratios to the two-point norm instead
        lyD = X.log_y(D); lN = E.log_two_point(D, 2*ts, m_grid=D, ts=ts)
        r1 = H.lse(lT) - H.lse(lyD); r2 = lref - lN - (2*(wa + wb + wc) - 2*D)*np.log(s)
        print(f"C={C} dy={dy:+.3f} 3e ratio amplitude/norm: replica pieces {np.exp(r1):.10f}   Step 3e code {np.exp(r2):.10f}")
    # (2) deep post-Page with equal levels (omega = 0): decoded = physical
    for p in (1, 2):
        lT = X.log_T(-0.347, p/2, p/2, 2.0 - p/2); ly = X.log_y(2.0)
        Cd, Cp, Cdeep, c1, _ = X.decode(lT, ly, ly, np.inf)
        print(f"C={C} omega=0 p={p}: C_deep/C_phys - 1 = {abs(Cdeep/Cp - 1):.1e}   C_dec(lam0=0)/C_phys - 1 = {abs(Cd/Cp - 1):.1e}   c1 = {c1:.12f}")
