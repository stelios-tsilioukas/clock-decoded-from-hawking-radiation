# A6 stopwatch split: independent re-derivation (agent "stopwatch-split")

The agent was unable to write this file, so it was transcribed from its final message. The code and raw output in this
folder are the agent's own.

## Q1. Classical reading
On a fixed background the cross-ratio χ is real, because the pulses sit symmetrically around the time-symmetric slice.
The χ block is therefore real, and the phase comes only from the prefactor:

R_cl(τ, y, ε) = arg sin((τ − ε/2 + iy)/2) + arg sin((τ + ε/2 + iy)/2), which tends to 2 atan[cot(τ/2) tanh(y/2)] as ε → 0.

- R_cl does not depend on Δ, h, ω or Δ_χ: the dimensions enter the phase only through Δ_b − Δ_a = ω, which divides out.
- Geometric meaning: R_cl is the clock's proper time from the event where it meets the ingoing ray sent at y = 0.
  At the horizon label R_cl = −τ exactly (checked in `geom_check.py`).
- Block, with s_ij = 2 sin((u_i − u_j)/2) and χ = s21 s43/(s31 s42):

  F_ab ∝ s21^{−Δa−h} s43^{−h−Δb} (s42/s41)^{Δa−h} (s41/s31)^{h−Δb} χ^{Δχ} ₂F₁(Δχ−Δa+h, Δχ+h−Δb; 2Δχ; χ)

## Exact formula
F_ab = ∫dμ(k_B)dμ(k_D) e^{−s[εk_B² + (2π−2τ)k_D²]} K_a(ℓ_A) K_b(ℓ_C) / g_{Δχ}(k_B, k_D)

- K_a = ∫dμ(k_i) g_{w2}(k_B,k_i) g_{w3}(k_D,k_i) H_a(k_i)
- H_a = ∫dμ(k_A) e^{−sℓk_A²} g_{w1}(k_A,k_i)
- Weights: w1 = (Δa+h−Δχ)/2, w2 = (h+Δχ−Δa)/2, w3 = (Δχ+Δa−h)/2
- Arc lengths: ℓ_A = τ + iy − ε/2 and ℓ_C = conj(ℓ_A)

## One-loop check
- Method: a one-loop Schwarzian expansion of the same block, with propagator
  ⟨e(u)e(0)⟩ = (1/2πC)[−(u−π)²/2 + (u−π) sin u + 1 + π²/6 + (5/2) cos u].
- Comparison at y = −0.3, Δ = 2:
  - one-loop coefficient: a = 0.15012;
  - exact C·(exact − R_cl) over C = 10–160, Richardson-extrapolated: 0.15011.

## Q2. Bare τ_s = π/4, Δ = 2, ω = 0.2, h = 1, ε = 0.05
Grid-converged to < 1e-11.

| C | κ₁ | label | exact | \|V\| | R_cl(κ=1) at the same y | R_cl(κ₁) |
|---|---|---|---|---|---|---|
| 10 | 1.190786 | H | −0.93874067 | 0.99170 | −0.96027 | −0.93590 |
| 10 | 1.190786 | 0.9 in | +0.85954513 | 0.99179 | +0.87892 | +0.85695 |
| 20 | 1.100593 | H | −0.86635891 | 0.99177 | −0.87610 | −0.86505 |
| 20 | 1.100593 | 0.9 in | +0.79088876 | 0.99185 | +0.79965 | +0.78970 |

## Q3. Fixed μ = 0.2: C·(exact − R_cl(κ₁))

| C, Δ | H | 0.9 in |
|---|---|---|
| 10, 2 | −0.0284539 | +0.0259752 |
| 20, 4 | −0.0283699 | +0.0259022 |
| 40, 8 | −0.0283611 | +0.0258949 |
| 80, 16 | −0.0283613 | +0.0258953 |

- The values converge, so the hypothesis holds: the O(1) backreaction is entirely a temperature shift, to about 1e−5.
- Split at the protocol point, measured from R_cl(κ = 1) at the same bare y:

| C | label | total | classical | quantum |
|---|---|---|---|---|
| 10 | H | 0.21529 | 0.24375 | −0.02845 |
| 20 | H | 0.19479 | 0.22097 | −0.02618 |

## Q4. Physical variables (k₁ = 20, τ_s′ = π/4, m = 2, horizon label)

| (ω, h, ε) | k₁(exact − cl)/\|cl\| |
|---|---|
| (0.2, 1, 0.05) | −0.033061 |
| (0.1, 1, 0.05) | −0.039437 |
| (0.05, 1, 0.05) | −0.042627 |
| (0.1, 0.5, 0.05) | −0.042272 |
| (0.05, 0.25, 0.05) | −0.046921 |
| (0.2, 1, 0.025) | −0.032838 |

- Fit: c₀ = −0.0516, c_ω = +0.0646, c_h = +0.0057.
- Interpreting c₀ at one loop: a = 0.14399, made of 0.18009 from the temperature shift and −0.03610 that is purely
  quantum and independent of Δ.
- That gives c₀ = −0.0454 at strict one loop and −0.0495 at finite C = 17.67, against the fitted −0.0516.

## Doubts
1. The cubic vertex and amputated gluing were taken as given; the one-loop check uses the same block structure.
2. The split uses m = Δ in the κ₁ map, which is confirmed at O(Δ/C). A different O(1/C) convention would move O(1)
   amounts between the classical and quantum columns.
3. "0.9 inner" was read as y = −0.9·y_H.
4. Sign convention: reading = arg(F₀₁)/ω with ψ₀ in the ket.
