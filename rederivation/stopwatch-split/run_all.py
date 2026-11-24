"""Production runs for questions 2-4. Usage: python3 run_all.py q2|q3|q4"""
import sys, json, time
import numpy as np
from sw import exact_F, R_cl, bare_to_phys, phys_to_bare, yH_of
from oneloop import a_coeff

tau0 = np.pi / 4
out = {}
which = sys.argv[1]


def labels_bare(C, tau, m):
    d = bare_to_phys(C, tau, m)
    k = d['kappa1']
    yH = yH_of(k * tau) / k
    return d, {'H': yH, '0.9in': -0.9 * yH}


if which == 'q2':
    eps, om, h, D = 0.05, 0.2, 1.0, 2.0
    for C in (10, 20):
        d, labs = labels_bare(C, tau0, D)
        for lab, y in labs.items():
            rows = []
            for dk, kx, kxb in [(0.2, None, None), (0.1, None, None), (0.05, None, None),
                                (0.1, C + 14 * np.sqrt(C) + 30, C + 14 * np.sqrt(C) + 330)]:
                t = time.time()
                r = exact_F(C, tau0, y, eps, D, om, h, dk=dk, kmax=kx, kmaxB=kxb)
                rows.append(dict(dk=dk, N=r['N'], NB=r['NB'], reading=r['reading'], absV=r['absV'],
                                 sec=time.time() - t))
                print(C, lab, rows[-1], flush=True)
            k = d['kappa1']
            out[f'C{C}_{lab}'] = dict(C=C, kappa1=k, k1=d['k1'], taup=d['taup'], y=y, label=k * y, conv=rows,
                                      cl1=R_cl(tau0, y, eps), clk=R_cl(k * tau0, k * y, k * eps),
                                      PT_a=a_coeff(tau0, y, eps, D, om, h))
            print(out[f'C{C}_{lab}'], flush=True)

elif which == 'q3':
    eps, om, h = 0.05, 0.2, 1.0
    for C, D, dk in [(10, 2, 0.1), (20, 4, 0.1), (40, 8, 0.1), (80, 16, 0.1), (10, 2, 0.05), (40, 8, 0.05),
                     (80, 16, 0.15), (160, 32, 0.15)]:
        d, labs = labels_bare(C, tau0, D)
        k = d['kappa1']
        for lab, y in labs.items():
            t = time.time()
            r = exact_F(C, tau0, y, eps, D, om, h, dk=dk)
            cl1 = R_cl(tau0, y, eps)
            clk = R_cl(k * tau0, k * y, k * eps)
            rec = dict(C=C, Delta=D, dk=dk, N=r['N'], NB=r['NB'], kappa1=k, y=y, reading=r['reading'],
                       absV=r['absV'], cl1=cl1, clk=clk, C_ex_m_clk=C * (r['reading'] - clk),
                       C_ex_m_cl1=C * (r['reading'] - cl1), C_clk_m_cl1=C * (clk - cl1), sec=time.time() - t)
            out[f'C{C}_D{D}_dk{dk}_{lab}'] = rec
            print(rec, flush=True)

elif which == 'q4':
    k1, taup, m = 20.0, np.pi / 4, 2.0
    d = phys_to_bare(k1, taup, m)
    C, k, tau = d['C'], d['kappa1'], d['tau']
    lab = yH_of(taup)
    y = lab / k
    out['map'] = {kk: float(v) for kk, v in d.items()}
    out['map']['label_H'] = lab
    out['map']['y_bare'] = y
    print(out['map'], flush=True)
    pts = [(0.2, 1, 0.05), (0.1, 1, 0.05), (0.05, 1, 0.05), (0.1, 0.5, 0.05), (0.05, 0.25, 0.05), (0.2, 1, 0.025)]
    for (om, h, ep) in pts:
        rec = {}
        for conv in ('phys', 'bare'):     # eps given in physical units (eps_bare = ep/k) or bare units
            eb = ep / k if conv == 'phys' else ep
            dk, dk2, kxb = (0.1, 0.05, None) if h >= 0.5 else (0.04, 0.03, C + 9 * np.sqrt(C) + 3 * m + 50)
            t = time.time()
            r = exact_F(C, tau, y, eb, m, om, h, dk=dk, kmaxB=kxb)
            r2 = exact_F(C, tau, y, eb, m, om, h, dk=dk2, kmaxB=kxb)
            cl = R_cl(taup, lab, k * eb)
            a = a_coeff(tau, y, eb, m, om, h)
            rec[conv] = dict(eps_bare=eb, reading=r['reading'], reading_half_dk=r2['reading'], absV=r['absV'],
                             cl_phys=cl, cl_bare=R_cl(tau, y, eb),
                             obs=k1 * (r['reading'] - cl) / abs(cl),
                             PT_reading=R_cl(tau, y, eb) + a / C,
                             PT_obs=k1 * (R_cl(tau, y, eb) + a / C - cl) / abs(cl), sec=time.time() - t)
            print((om, h, ep), conv, rec[conv], flush=True)
        out[f'{om}_{h}_{ep}'] = rec

json.dump(out, open(f'results_{which}.json', 'w'), indent=1, default=float)
