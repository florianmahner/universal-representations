### Factorization

Fit non-negative embeddings from each model's RBF similarity matrix, then select the bandwidth using cross-seed stability and explained variance.

#### Fit and select

Run from the repository root:

```bash
python -m experiments.factorization.fit.run_grid
python -m experiments.factorization.bandwidth_selection.compute_selection
python -m experiments.factorization.bandwidth_selection.plot
```

The rank, bandwidth grid, and seeds are set in [`config.yaml`](../../config.yaml). To fit another rank, append `factorization.rank=100` to the fitting and selection commands.

#### Inputs and outputs

Features are read from `data/features/<dataset>/*.npy`. Embeddings are written to `data/embeddings/<dataset>/`; the selected bandwidths and seeds are saved in `data/meta/<dataset>/bandwidth_selection_rank_<rank>.csv`.
