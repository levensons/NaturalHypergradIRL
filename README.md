# NaturalHypergradIRL

CBSCFD sketch compression supports `baseline` (default) and `qr_only`. To
enable QR compression in an IRL configuration, set:

```yaml
fisher:
  mode: sketch
  cbscfd_variant: qr_only
```

Keep the other Fisher settings, including `sketch_size` and positive
`fisher_reg`, in the configuration. The selected variant is logged as an MLflow
parameter. Direct callers can pass `variant="qr_only"` to `CBSCFD`, or
`cbscfd_variant="qr_only"` to `FisherNHD`.

For a buffer `A` of shape `(2*m, dim)`, QR compression factors `A.T` and runs
SVD on the smaller `R.T` matrix. It preserves the CBSCFD shrinkage and inverse
solve, up to floating-point roundoff. `svd_time` measures only the smaller SVD;
QR and sketch reconstruction are included in `extend_time` and `other_time`.

Numerical tests compare the variants' sketch Gram matrices, regularization,
and inverse solves, and exercise streaming and materialized Fisher paths:

```bash
python -m pytest tests/algorithms
```

On the HSE cluster, run tests inside a short Slurm job with one CPU, no GPU,
and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
