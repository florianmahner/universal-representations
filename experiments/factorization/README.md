# Factorization

RBF similarity kernel per model, symmetric NMF (S ≈ WWᵀ) over a bandwidth × seed grid, then per-model bandwidth selection. Sits between `experiments/feature_extraction` and `experiments/universality`.

| Analysis | Reproduces | Run |
|---|---|---|
| [`fit/`](fit/) | SRF embeddings for all models (Sec. 3.3) | `python -m experiments.factorization.fit.run_grid` |
| [`bandwidth_selection/`](bandwidth_selection/) | Per-model bandwidth maximizing the harmonic mean of cross-seed stability and explained variance (Appendix, bandwidth selection) | `python -m experiments.factorization.bandwidth_selection.compute_selection`, then `...bandwidth_selection.plot` |

Run them in this order:

```bash
python -m experiments.factorization.fit.run_grid              # writes data/embeddings/things22k/<model>/rank_50/sigma_*/seed_*/
python -m experiments.factorization.bandwidth_selection.compute_selection   # writes data/meta/things22k/bandwidth_selection_rank_50.csv
python -m experiments.factorization.bandwidth_selection.plot  # supplementary bandwidth figure
```

The grid (rank, bandwidths, seeds) is pinned in the root `config.yaml`; override for other ranks, e.g. `factorization.rank=100`. Inputs are `data/features/things22k/*.npy` from `experiments/feature_extraction`; the fitted `w.npz` files and the bandwidth-selection CSV are what `experiments/universality` consumes.
