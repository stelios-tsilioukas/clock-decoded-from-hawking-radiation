# A1 + A2 one-sided JT: independent re-derivation (agent "jt-oneside")

The agent was unable to write this file, so it was transcribed from its final message. The code and raw output in this
folder are the agent's own.

## A1: classical (all checks to machine precision)
- **Phase:** arg A_cl = ω·τ(y), with τ(y) = 2 arctan[cot(τ_s/2) tanh(y/2)].
- **Modulus:** |A_cl| = (2 sin τ_s)^(−2Δ̄) R^h, with R = dτ/dy = sin τ_s/(cosh y − cos τ_s).
- **Geometry:** in AdS₂ embedding coordinates the clock is X₀ = cos σ/sin τ_s, X₁ = cot τ_s cos σ, T = sin σ. The
  ingoing ray from y meets the clock at σ(y) = π/2 + τ(y), so arg A = ω[σ(y) − σ(0)].
- **Independent null-ray check:** phase agrees to 1e−14, modulus to 6e−16.
- **Horizon:** σ_h = π/2 − τ_s, which gives y_H = ln cos τ_s = −2 artanh tan²(τ_s/2) (−0.346574 at π/4). The proper time
  from the horizon to the y = 0 event is τ_s = 0.785398.
- **Classical tick rates R_cl:**

| label | y_H | 0 | −0.9y_H | y_H−1 | y_H−2 |
|---|---|---|---|---|---|
| R_cl | 2.000000 | 2.414214 | 2.067958 | 0.525701 | 0.154877 |

## A2: exact disk formula
**Regions:**
- region 1: arc length ℓ₁ = τ_s + iy, momentum k₁
- region 2: arc length ℓ₂ = τ_s − iy, momentum k₂
- region 3: arc length ℓ₃ = 2π − 2τ_s, momentum k₃
- the inner triangle, momentum q

**Formula:**
- M_p = ∫dμ(q) F_{p/2}(q;ℓ₁) F_{p/2}(q;ℓ₂) F_{m−p/2}(q;ℓ₃)
- F_w(q;ℓ) = ∫dμ(k) e^(−ℓk²/2C) g_w(k,q)/(2C)^(2w)
- ⟨R^p⟩ = M_p/(M_0 (2C)^p)

**Numerics:**
- trapezoid rule; going from h = 0.04 to 0.02 changes the results by ≤ 2e−11;
- C = 10…160, m = 0.75, 1.5, 2, 3;
- two-point check at O(1/C): 1.5237, against 1.523670 at one loop.

## Results at τ_s = π/4, m = 3

| label | y | C·Var/⟨R⟩², C=20 | C=40 | C → ∞ | C(⟨R⟩/R_cl−1), C=20 | C=40 |
|---|---|---|---|---|---|---|
| y_H | −0.346574 | 0.157077 | 0.156640 | 0.156198 | −0.361326 | −0.364238 |
| 0 | 0 | 0.131384 | 0.131077 | 0.130766 | −0.303950 | −0.306064 |
| −0.9y_H | +0.311916 | 0.152192 | 0.151782 | 0.151367 | −0.350468 | −0.353220 |
| y_H−1 | −1.346574 | 0.522811 | 0.518257 | 0.513612 | −1.113715 | −1.139097 |
| y_H−2 | −2.346574 | 1.341225 | 1.314087 | 1.286259 | −2.461880 | −2.585590 |

**Mean shift.** After 1/C → 0, C(⟨R⟩/R_cl−1) is exactly linear in m: a₀ + a₁ m.

| label | a₀ | a₁ |
|---|---|---|
| y_H | 0.055772 | −0.140987 |
| 0 | 0.046651 | −0.118284 |
| −0.9y_H | 0.054038 | −0.136681 |
| y_H−1 | 0.185493 | −0.450330 |
| y_H−2 | 0.47462 | −1.06517 |

**One-loop check.** The Schwarzian propagator (1/2πC)Σ_{|n|≥2} e^(inu)/(n²(n²−1)) reproduces all of these limits to
6–7 digits:
- variance: 0.1561983, 0.1307665, 0.1513671, 0.5136124, 1.2862587;
- a₀: 0.0557718, 0.0466514, 0.0540381, 0.1854935, 0.4746196.

## Doubts
1. "0.9 inner" was taken as y = −0.9·y_H = +0.3119.
2. τ(0) is measured from where the y = 0 ray meets the clock, which is inside the horizon; this is not the clock's
   time-symmetric point.
3. Coefficient (ii) depends on operator normalisation, here (2C)^(−2w). Var and a₁ do not.
4. Var needs m > 1; the m → 0 extrapolation uses m ≥ 0.75, which is safe because the result is exactly linear in m.
5. The sign of (ii) at m = 3 depends on whether bare or physical labels are used. The bare↔physical map was not tested.
6. The phase sign assumes u = iy with y running to the future.
7. No closed form was found beyond the propagator expressions.
