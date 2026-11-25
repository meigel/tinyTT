# Hybrid discrete posteriors

`tinytt.hybrid` handles posteriors over discrete configurations `s = (s_1, ..., s_d)` of the form

    p(s) ∝ C(s) · A(s),

where the *computation network* `C` is chain-structured (unary factors `u_m(s_m)` and pair
factors `W_m(s_m, s_{m+1})`: a Markov prior, hard 0/1 rules between neighbours, per-site
evidence) and the *activation network* `A` is a TT — typically a cross approximation of a
likelihood, possibly multiplied by the TT of a global formula.

## Formulas as automata

A formula that a deterministic automaton can check while reading `s_1, ..., s_d` has an exact TT
whose ranks are bounded by the number of automaton states.  Counting is the standard example:

```python
import numpy as np
import tinytt as tt
from tinytt.hybrid import automaton_tt, counting_tt, chain_contract, born_contract, tempered_cross
import tinytt.backend as tn

# at most K = 2 of the d = 8 sites take a "clay" state (states 4, 5 of 6)
clay = np.array([False, False, False, False, True, True])
phi = counting_tt(clay, 8, 2)
print(phi.R)                    # automaton bond dimension K + 2: [1, 4, 4, 4, 4, 4, 4, 4, 1]
print(phi.round(1e-12).R)       # exact ranks min(k, K, d-k, d-K) + 1: [1, 2, 3, 3, 3, 3, 3, 2, 1]
```

`automaton_tt(delta, accept, length)` builds the TT for any transition table; `-1` marks the dead
state and a `(length, Q, n)` table makes the automaton position dependent.

## Exact contraction

`chain_contract(x, unary, pair)` returns the normaliser and all single-site marginals of
`C · x` in one left/right sweep, without truncation:

```python
d, n = 8, 6
rng = np.random.default_rng(0)
unary = [rng.random(n) for _ in range(d)]
pair = [rng.random((n, n)) for _ in range(d - 1)]
x = tt.random([n] * d, 3)
A = (x * x) * phi                              # nonnegative activation times the formula
Z, marginals = chain_contract(A, unary, pair)
print(round(float(tn.to_numpy(marginals[0]).sum()), 12))   # 1.0
```

A truncated TT of a nonnegative function can take negative values, and so can the contracted
"marginals".  `born_contract(g, unary, pair)` contracts `C · g²` instead, which is a probability
distribution for any real TT `g`; for `g ≈ sqrt(A)` the total-variation error is at most
`2 ‖√C (√A − g)‖ / ‖√(C A)‖`.

```python
Zb, mb = born_contract(x, unary, pair)
print(min(float(tn.to_numpy(m).min()) for m in mb) >= 0)   # True
```

## Tempered cross

A peaked likelihood is numerically zero on most index sets a cold-started cross proposes.
`tempered_cross(f, N, betas)` approximates `f**beta` for increasing `beta`, warm-starting each
stage from the previous one, and records the `maxvol_converged` flag of every stage:

```python
center = np.array([2, 5, 1, 6, 3])
def f(I):
    I = tn.to_numpy(I).astype(float)
    return tn.tensor(np.exp(-4.0 * ((I - center) ** 2).sum(axis=1)), dtype=tn.float64)

y, info = tempered_cross(f, [8] * 5, betas=(0.1, 0.4, 1.0), eps=1e-10, nswp=6)
print(len(info["stages"]), info["maxvol_converged"])
```
