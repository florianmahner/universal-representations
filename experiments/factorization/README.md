# Factorization

Fit SRF embeddings from extracted features, then select the bandwidth using cross-seed stability and explained variance. Run from the repository root:

```bash
python -m experiments.factorization.fit.run_grid
python -m experiments.factorization.bandwidth_selection.compute_selection
python -m experiments.factorization.bandwidth_selection.plot
```

The fitting grid is set in [`config.yaml`](../../config.yaml). Inputs are `data/features/<dataset>/*.npy`; outputs are `data/embeddings/<dataset>/` and `data/meta/<dataset>/bandwidth_selection_rank_<rank>.csv`.
