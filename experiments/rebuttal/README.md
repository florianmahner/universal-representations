# Additional analyses

These analyses use the embeddings and scores from [factorization](../factorization/) and [universality](../universality/).

| Directory | Analysis |
|---|---|
| [universality_across_layers](universality_across_layers/) | Early, middle, and late layers; untrained networks; layer-wise human ratings |
| [baselines](baselines/) | Pixels, pixel PCA, Gabor features, and random noise |
| [ipcl](ipcl/) | Training-data domain |
| [alignnet](alignnet/) | Models post-trained for human alignment |
| [greedy_matching](greedy_matching/) | Greedy versus Hungarian matching |
| [categorical_control](categorical_control/) | Category consistency as a control for predictivity |
| [label_grounding](label_grounding/) | Semantic and visual predictors of dimensions |
| [llm_dimension_descriptions](llm_dimension_descriptions/) | Model-generated descriptions and human-label comparisons |
| [predictions_across_depth](predictions_across_depth/) | Neural and behavioral prediction across layers |

Run modules from the repository root, for example:

```bash
python -m experiments.rebuttal.categorical_control.run
```

For the depth analysis, run these stages in order:

```bash
python -m experiments.rebuttal.universality_across_layers.layer_selection.make_spec
python -m experiments.rebuttal.universality_across_layers.feature_extraction.extract_full
python -m experiments.rebuttal.universality_across_layers.factorization.factorize
python -m experiments.rebuttal.universality_across_layers.factorization.select_sigma
python -m experiments.rebuttal.universality_across_layers.scoring.score
python -m experiments.rebuttal.universality_across_layers.factorization.artifacts
```
