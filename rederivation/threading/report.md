# A3 threading vs caps: independent re-derivation (agent "threading")

The agent was unable to write this file, so it was transcribed from its final message. The code and raw output in this
folder are the agent's own.

## Setup and brane rule
- **Boundary.** Each replica's boundary is u ∈ [−π, π), closed by the brane at u = ±π. The clock is created at −τ_s and
  annihilated at +τ_s, so each insertion sits at distance π − τ_s from the brane.
- **Connected two-replica disk** (total boundary length 4π). The points in order:
  B₁ (0), c₁⁺ (π−τ_s), c₁⁻ (π+τ_s), B₂ (2π), c₂⁺ (3π−τ_s), c₂⁻ (3π+τ_s).
- **Brane weight.** Each brane segment on a region with momentum k contributes y_μ(k) = |Γ(μ−½+ik)|²/Γ(μ−½)² (PSSY).
  The agent takes μ → ∞ at fixed C, so y → 1.
- **Excluded pairing.** The crossing (non-planar) pairing is excluded.

## Integrals
H_n(k; ℓ) = ∫dμ(a) y(a)ⁿ e^{−s_β ℓ a²} g_Δ(a, k)

- Caps: Z₂^C = ∫dμ(k) y(k)² e^{−s_β·4(π−τ_s)k²} [H₀(k; 2τ_s)]²
- Threading: Z₂^T = ∫dμ(k) e^{−s_β·4τ_s k²} [H₁(k; 2(π−τ_s))]²

**Exact symmetry at μ = ∞:** Z^T(τ_s) = Z^C(π − τ_s), so the log-ratio is odd about τ_s = π/2.

## Probe limit
lim_{C→∞} log(Z₂^T/Z₂^C) = 4Δ log tan(κ*τ_s), with κ* = ½ at μ = ∞, so the limit is 4Δ log tan(τ_s/2).

**1/C correction:**
δ = [Δ²(Q(π−τ_s) − Q(τ_s)) + Δ(L(π−τ_s) − L(τ_s))]/C

with T = cot(σ/2) and ℓ = 2σ:
- Q(σ) = 4T(1 − ℓ/π) + T²ℓ²/(2π) − ℓT²
- L(σ) = 2T(1 − ℓ/(2π)) − ℓ(1+T²)/2 + (1+T²)ℓ²/(8π)

At τ_s = π/4 this gives (1.007Δ² − 0.447Δ)/C. After subtracting it, the residual scales as 1/C².

## Exact table, log(Z^T/Z^C) at μ = ∞

Each entry reads C=10 / C=40 / C=160 / probe.

| τ_s/π | Δ=0.5 | Δ=1 | Δ=2 | Δ=4 |
|---|---|---|---|---|
| 0.25 | −1.7600 / −1.7620 / −1.7626 / −1.7627 | −3.4714 / −3.5116 / −3.5220 / −3.5255 | −6.7564 / −6.9738 / −7.0315 / −7.0510 | −12.8242 / −13.7542 / −14.0131 / −14.1020 |
| 0.375 | −0.8056 / −0.8062 / −0.8063 / −0.8064 | −1.5881 / −1.6065 / −1.6112 / −1.6128 | −3.0879 / −3.1896 / −3.2165 / −3.2256 | −5.8487 / −6.2876 / −6.4094 / −6.4512 |
| 0.5 | 0 | 0 | 0 | 0 |

The rows at τ_s/π = 0.625 and 0.75 are the 0.375 and 0.25 rows with the sign flipped.

**Checks:** quadrature step, momentum window and margin are stable to the 6th decimal; the special functions agree with
mpmath to about 1e−14.

**Finite tension (μ = C, Δ = 1):**
- The saddle equation π(1 − 2κ*) = 2 atan(κ*) gives κ* = 0.3834.
- At τ_s = π/2 the exact ratio is −1.5526, −1.5116, −1.5018 for C = 10, 40, 160; the probe value is −1.4985.
- The crossover sits at τ_s = π/(4κ*) = 0.652π, while the single-replica horizon is at 0.635π.

## Interpretation
- Caps dominate for τ_s < π/2 and threading for τ_s > π/2.
- τ_s = π/2 is where the clock geodesic crosses the bifurcation point.
- For n replicas the two pairings tie at κ_n τ_s = π/(2n).

## Doubts and assumptions
- **How sharp the switch is.** At fixed Δ the suppression is exponential in Δ, not in C. For Δ = 0.5 at π/4 the losing
  pairing is still about 17% of the winner, so a sharp switch needs 1 ≪ Δ ≪ C.
- **"Large tension"** was read as μ → ∞ at fixed C.
- **Normalisation:** only the connected two-replica geometry was used; common factors cancel in the ratio.
- **Clock field:** a single complex field, with the crossing pairing excluded.
- **Backreaction:** the bare τ_s is identified with τ_s′.
