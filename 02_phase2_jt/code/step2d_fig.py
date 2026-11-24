import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('02_phase2_jt/results/step2d_runs.json'))
A = json.load(open('02_phase2_jt/results/step2d_analysis.json'))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
cols = {0.393: 'C0', 0.785: 'C1', 1.178: 'C2'}
for r in R:
    if r['k'] == 0.05 and abs(r['kappa'] - 0.472) < 1e-2:
        ts = np.arcsin(2*r['kappa']/r['phi_s']); c = cols[round(ts, 3)]
        ax[0].plot(r['T2'], r['loop_static'], '-', color=c, label=fr'static, $\tau_s={ts/np.pi:.3f}\pi$')
        ax[0].plot(r['T2'], r['loop_evap'], 'o', color=c, ms=4)
ax[0].set_xlabel(r'station proper time $T_2$  [$\ell$]'); ax[0].set_ylabel(r'$\Delta\tau_{loop}$  [$\ell$]')
ax[0].set_title('Twin loop: evaporating (dots, $k=0.05$, $\\kappa=0.47$)\nvs static at instantaneous $\\kappa$ (lines)', fontsize=10)
ax[0].legend(fontsize=8)
names = ['drop→φ=0 − π/2', 'horizon→φ=0 − τ_s', '2κ(Y_end − y_radar)', 'max |loop dev|']
mk = ['o', 's', '^', 'D']
for key, tab in A['coefficients'].items():
    tab = np.array(tab); c = cols[float(key)]
    for j in range(4):
        sel = tab[:, 1] > 0.7
        ax[1].semilogx(tab[sel, 0], np.abs(tab[sel, 2 + j]), mk[j] + '-', color=c, ms=4, lw=.8,
                       label=names[j] if key == '1.178' else None)
ax[1].set_yscale('log'); ax[1].set_xlabel('k'); ax[1].set_ylabel(r'|deviation| / ($k/\kappa$)')
ax[1].set_title('All deviations collapse onto $k/\\kappa=c/6(S-S_0)$\n(colour = τ_s: blue π/8, orange π/4, green 3π/8; κ=0.78)', fontsize=10)
ax[1].legend(fontsize=7, loc='center right')
E = A['endurance']; kk = np.array([e['k'] for e in E]); tx = np.array([e['tau_x'] for e in E]); pr = np.array([e['pred'] for e in E])
ax[2].semilogx(kk, tx/pr, 'o'); ax[2].axhline(1, color='k', lw=.8)
ax[2].set_xlabel('k'); ax[2].set_ylabel(r'$\tau_\times$ numerical / formula')
ax[2].set_title(r'QES-locus crossing: $\tau_\times=\frac{\eta}{1+\eta}\frac{k}{\kappa}\tan\tau_s$' '\n' r'($\tau_s=\pi/4$; correction $O(k/\kappa)$)', fontsize=10)
plt.tight_layout(); plt.savefig('02_phase2_jt/results/fig_step2d_clock.png', dpi=150)
