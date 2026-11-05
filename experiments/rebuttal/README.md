### Additional analyses

Controls and extensions using the embeddings and scores from [factorization](../factorization/) and [universality](../universality/).

#### Network depth

[`universality_across_layers`](universality_across_layers/) compares early, middle, and late layers. Run these stages from the repository root:

```bash
python -m experiments.rebuttal.universality_across_layers.layer_selection.make_spec
python -m experiments.rebuttal.universality_across_layers.feature_extraction.extract_full
python -m experiments.rebuttal.universality_across_layers.factorization.factorize
python -m experiments.rebuttal.universality_across_layers.factorization.select_sigma
python -m experiments.rebuttal.universality_across_layers.scoring.score
python -m experiments.rebuttal.universality_across_layers.factorization.artifacts
```

The same directory contains untrained-network and layer-wise human-rating analyses. [`predictions_across_depth`](predictions_across_depth/) relates layer-wise universality to neural and behavioral prediction.

#### Baselines and model controls

- [`baselines`](baselines/): pixels, pixel PCA, Gabor features, and random noise.
- [`ipcl`](ipcl/): training-data domain with architecture and objective fixed.
- [`alignnet`](alignnet/): models post-trained for human alignment.
- [`greedy_matching`](greedy_matching/): greedy versus Hungarian matching.

#### Dimension content

[`label_grounding`](label_grounding/) tests semantic and visual predictors of dimensions. [`llm_dimension_descriptions`](llm_dimension_descriptions/) generates descriptions and compares them with human labels.

[`categorical_control`](categorical_control/) tests whether category consistency accounts for neural and behavioral predictivity:

```bash
python -m experiments.rebuttal.categorical_control.run
```
