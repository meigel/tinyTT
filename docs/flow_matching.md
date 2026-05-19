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

For high-dimensional curved banana cases, `--baseline-degree` fits a fixed
least-squares polynomial baseline to the flow-matching velocity

```text
u_t(z_t) = x1 - x0,    z_t = (1 - t) x0 + t x1.
```

The trained TT velocity is then a residual correction. Enable
`--baseline-interaction-pairs first`, `adjacent`, or `all` when the map
contains rotated quadratic structure: it adds selected `x_i x_j t^q` features,
which are important when the interpolation coordinates mix the original banana
coordinates. Prefer sparse choices in high dimension; `all` scales like
`d(d-1)/2`. The older `--baseline-interactions` flag is kept as an alias for
`--baseline-interaction-pairs all`.

With a baseline active, `--baseline-bias-mode` controls the learnable residual
bias initialization. `residual` initializes from the residual mean and gives a
clean baseline at startup. `total` initializes from the total displacement; this
can be a useful warm start when residual training is expected to make a larger
correction, but it double-counts the low-order baseline initially. `zero` leaves
the residual bias at zero.

The current practical d=20 shifted rotated banana configuration is:

```bash
python -m examples.flow_matching_suite \
  --dims 20 --cases banana --rank 10 --output-rank 20 --epochs 500 \
  --train 2048 --batch 256 --eval 128 --metric-points 128 \
  --metric-every 100 --sinkhorn-every 0 --loss-every 10 --device CUDA \
  --learnable-bias --no-cutoff --baseline-degree 3 \
  --baseline-time-degree 2 --baseline-samples 4 \
  --baseline-interaction-pairs adjacent --baseline-bias-mode total \
  --banana-curvature 1.5 --banana-angle 45 --banana-shift 0.5
```

On the tracked CUDA run this reduced energy from about `0.540` to `0.00153`
and Sinkhorn from about `2.584` to `0.0109`.

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
