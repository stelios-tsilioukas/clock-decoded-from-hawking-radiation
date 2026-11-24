"""
Step 3i (open point 3) -- the decoded two-time interval of the frozen stopwatch and the decoder's convention.

(1) Decoder convention (Yoshida-Kitaev U*; Phase 1 Lemma 2).  In the PSSY model the Petz map built from R depends on
    the code states only through Tr_B(|Psi_a><Psi_b|), and Tr_B[(U_B x 1) X (U_B^dag x 1)] = Tr_B X for any unitary U_B
    on the black hole.  Hence Bob's model of the black-hole evolution (a time offset delta t, or any B-unitary applied to
    his reference states) drops out of the decoded register state identically.  What remains is the phase convention of
    the logical basis (relative phase alpha of the reference operators chi_0, chi_1): it multiplies rho_01 by e^{i alpha}
    and shifts every reading by alpha/omega, independently of the stop time y.  Checked numerically below: the decoded
    interval between two stop events is unchanged when the chi_1 reference is rephased.
(2) Decoded two-time interval  Delta_dec(y1, y2) = [arg rho01_dec(y2) - arg rho01_dec(y1)]/omega, same decoder for both
    stop events (two runs, or two registers), compared with the physical one-sided interval and the classical proper time
    between the two absorption events.  The twin-loop phase of Phase 2 is omega [Delta tau_C1 - Delta tau_C2]; the
    boundary (C2) interval is classical and exact, so the decoded loop phase is omega Delta_dec minus a known number.
"""
import numpy as np, json, sys
OUT = '03_phase3_quantum/results/'

def intervals(path, pairs):
    res = json.load(open(path)); out = []
    for r in res:
        C, xs, rows = r['C'], r['xs'], r['rows']
        print(f"\nC = {C}, tau_s = {r['tau_s']/np.pi:.2f} pi (island clock)")
        for a, b in pairs:
            A, Bb = rows[a], rows[b]
            cl = Bb['classical'] - A['classical']; ph = Bb['phys'][1] - A['phys'][1]
            line = dict(C=C, pair=[a, b], classical=cl, physical=ph, decoded={}, decoded_canonical={})
            print(f"  interval {a} -> {b}: classical {cl:+.5f}  physical (one-sided) {ph:+.5f}")
            for i, x in enumerate(xs):
                de = Bb['dec_exact'][i][1] - A['dec_exact'][i][1]; dc = Bb['dec_canonical'][i][1] - A['dec_canonical'][i][1]
                line['decoded'][str(x)] = de; line['decoded_canonical'][str(x)] = dc
                print(f"     x = {str(x):>5}: decoded {de:+.5f} (canonical {dc:+.5f})   decoded/physical - 1 = {de/ph - 1:+.5f}   C x (decoded - physical) = {C*(de - ph):+.4f}")
            out.append(line)
    return out

def convention_check(path, alpha=0.7, om=0.2):
    """rephasing the chi_1 reference by alpha shifts each decoded reading by alpha/omega (mod 2 pi/omega) and leaves intervals unchanged."""
    res = json.load(open(path)); worst = 0.0
    for r in res:
        rows = r['rows']; names = list(rows)
        for i in range(len(r['xs'])):
            rd = {nm: rows[nm]['dec_exact'][i][1] for nm in names}
            sh = {nm: np.angle(np.exp(1j*(om*rd[nm] + alpha)))/om for nm in names}
            for a in names:
                for b in names:
                    d0 = np.angle(np.exp(1j*om*(rd[b] - rd[a])))/om; d1 = np.angle(np.exp(1j*om*(sh[b] - sh[a])))/om
                    worst = max(worst, abs(d1 - d0))
    print(f"\nconvention check: max |interval(rephased) - interval| = {worst:.1e}  (single readings shift by alpha/omega = {alpha/om:.3f})")
    return worst

if __name__ == "__main__":
    paths = sys.argv[1:] or [OUT + 'step3i_island_C10.json']
    pairs = [('dy=-2', 'dy=-1'), ('dy=-1', 'dy=+0'), ('dy=+0', 'dy=+1'), ('dy=-2', 'dy=+1')]
    allout = []
    for p in paths:
        allout += intervals(p, pairs); convention_check(p)
    json.dump(allout, open(OUT + 'step3i_interval.json', 'w'), indent=1, default=float)
