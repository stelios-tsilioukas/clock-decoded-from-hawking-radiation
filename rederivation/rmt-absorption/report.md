# A4 RMT absorption test: independent re-derivation (agent "rmt-absorption")

The agent was unable to write this file, so it was transcribed from its final message. The tables are in
`data/tables_main.md` and `data/tables_finite_size.md`; the code and data in this folder are the agent's own.

## Summary
- **Monte Carlo against planar.** The MC agrees with the agent's own planar (large-L) prediction to 0.3–0.7% post-Page,
  for both the own and the unprojected model.
- **The other-placement term:**
  - size: O(1), about −0.25 × own; it saturates by k ≈ 2L and is not suppressed by L;
  - phase: essentially real and negative, arg 177–180°;
  - y-dependence: phase constant within 1°, magnitude falls only about 10% for y = 0 → 1.35.
- **Run settings:** 24 samples per regime; errors are 1 s.e.m.; numbers are ε-odd parts at ε = 0.05, not divided by ε.

## Q1. Own-placement model
The ε-odd part contains exactly one W, whose V pairs with Φ's V†. W's O₀ can pair two ways:
- with Φ's O₀† — the own placement;
- with an O₀† inside the Petz kernel — the other placement.

Using an independent Õ in both W and Φ leaves only the own pairing, with every other contraction weight unchanged. So at
planar level the own model is the own placement, and other = unprojected − own.

## Q2. Planar formula, all k
**Own placement:**

ρ₁₀^own = (ε/2)(N₀N₁)^{-1/2} Σ_{αβγ} w_{αβγ}(y) m_α 𝒯_{αγ}

- m_α = (2/π)∫ds s^{-1/2}(1 − s g_{0α}(s))
- 𝒯_{αγ} = (1/π)∫ds s^{-1/2} g_{1α}(s) x_γ(s)

**Dyson (linear-pencil) system.** This is PSSY's resolvent with the brane loop ω resolved into energy blocks.
- p_γ = Σ_e n_e σ²_{eγ} g_e
- ω = 1/(1 + Σ_γ L_γ p_γ/(1 + kωp_γ))
- x_γ = kω/(1 + kωp_γ)
- g_e = 1/(s + Σ_γ L_γ σ²_{eγ} x_γ)

**Unprojected model.** The same formula holds with an extra factor s·g_{0β}(s) in 𝒯, so the other-placement term is
−(…)∫ g_{1α}[1 − s g_{0β}] x_γ.

**Which energies are shared:**
- E_α (after the signal): shared, which produces m_α.
- E_γ (brane/preparation): shared across all replicas through the brane loop.
- E_β (between the signal and the clock line): private to Φ in the own placement. In the other placement it is shared,
  and pinned to Ψ₀'s output support.

## Q3. Results at y = 0.9, k = 20000 (MC, with planar in brackets)

| regime | unprojected | own | other | ρ₀₀ | ⟨Ψ₁\|Φ⟩ |
|---|---|---|---|---|---|
| thread | .00381−.00264i [.00382−.00264i] | .00536−.00272i [.00538−.00272i] | −.00155+.00008i [−.00156+.00008i] | 0.9703(2) | .00553+.00259i |
| thread2 | .00355−.00240i [.00355−.00239i] | .00481−.00239i [.00485−.00240i] | −.00127−.00001i [−.00130] | 0.9897(1) | .00511+.00235i |

The full k-tables (k = 48, 300, 1600, 8000, 20000) are in `data/tables_main.md`.

- **|other/own| as k grows (k = 48 → 300 → 1600 → 20000):**
  - thread: 0.133 → 0.236 → 0.254 → 0.258
  - thread2: 0.138 → 0.213 → 0.234 → 0.236
- **Finite L.** For L = 450, 900 and 1800 at fixed k/L, the deviations are consistent with zero at 0.5% or better.
- **Mechanism.** The other term sits where E_β ≈ E_α, so the signal phase e^{iy(E_α−E_β)} ≈ 1. Effective energy
  transfer: about 0.53 for the own term, against about 0.06 / 0.005 for the other.

**y-scan (thread2, k = 8000):**

| y | own | other |
|---|---|---|
| 0 | +.00601 | −.00133 |
| 0.45 | +.00569−.00134i | −.00131 |
| 0.9 | +.00480−.00239i | −.00126 |
| 1.35 | +.00362−.00294i | −.00119 |

**Net, deep post-Page:**
- unprojected ≈ 0.75 e^{−i9.5°} × ⟨Φ|Ψ₁⟩;
- own ≈ 0.96–0.99 × ⟨Φ|Ψ₁⟩.

## Checks
- The Bartlett sampler normalisation gives 1.0001.
- τ^{-1/2} computed via eigh agrees with the polar factor to 1e−15.
- The Dyson block solutions match sample averages.
- The factorised own formula reproduces the direct sample value to 0.2%.
- Grid effects are below 1e−6.
- The ε-even part of ρ₁₀ averages to zero, as symmetry requires.

## Doubts
1. **Reading of F.** The agent took one f_n per ket, so norms carry f2. The literal "F|B_i⟩" with f already inside would
   carry f2² and shift all numbers by about 10%; the conclusions survive. The `fpow=2` option switches to it.
2. **ε scaling.** Odd parts are not divided by ε; O(ε³) contamination is about 0.25%.
3. **Normalisation of Φ.** N₀ from O₀ was used in both models.
4. **Sign of y.** As in the spec; the opposite convention flips the phases.
5. **Scope of the Õ model.** It isolates the own placement only at O(ε) and at planar level.
6. **ρ₀₀.** No planar prediction was made for it.
