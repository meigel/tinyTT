# Flow Matching Benchmarks

The maintained benchmark entry point is `examples/flow_matching_suite.py`.
It trains paired conditional flow-matching maps with a time-dependent
functional TT velocity field.

## Metrics

The suite records energy distance at each `--metric-every` checkpoint.
Sinkhorn divergence is more expensive, so `--sinkhorn-every` controls its
training-time cadence separately. A value of `0` computes Sinkhorn only at the
first and final metric checkpoints, while final summary metrics always include
Sinkhorn.

Generated benchmark outputs live under `plots/` and `results/`; these are
ignored by git. Commit only curated benchmark artifacts that are intended to be
reviewed as documentation.

## Parametric PDE Direction

For parametric PDE flows, keep the data interface paired:

```text
(x0, mu) -> x1
```

Here `x0` is the state or coefficient representation, `mu` is the parameter
vector, and `x1` is the target state. The velocity field should receive both
state and parameter features, but rollout should update only the state:

```text
dz/dt = v_theta(z, t, mu),    mu fixed.
```

Benchmark metrics should include distributional quality, such as energy and
Sinkhorn, plus PDE-specific diagnostics: residual norm, boundary-condition
violation, conservation error, or rollout-time constraint violation. Those
metrics should be added as separate hooks rather than folded into the synthetic
banana and Gaussian-mixture benchmarks.
