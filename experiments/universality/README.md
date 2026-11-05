### Universality

Measure how consistently dimensions recur across models, then relate these scores to model properties, dimension content, and biological alignment.

#### Compute scores

Run from the repository root after fitting the model embeddings:

```bash
python -m experiments.universality.metric.compute.run
```

#### Validate the measure

[`metric/validate`](metric/validate/) contains null calibration, within-model stability, rank and model-set sensitivity, and comparisons with ObjectNet, CKA, and ridge prediction. Each analysis has its own `run.py` or `plot.py`.

#### Model properties and dimension content

[`controlled_comparisons`](controlled_comparisons/) tests architecture, objective, training data, model size, and accuracy:

```bash
python -m experiments.universality.controlled_comparisons.run
```

[`dimensions`](dimensions/) contains category consistency, sparsity, reconstruction importance, and clustering analyses.

#### Neural and behavioral prediction

Start with [`macaque_encoding`](downstream_predictions/macaque_encoding/) and [`things_behavior`](downstream_predictions/things_behavior/):

```bash
python -m experiments.universality.downstream_predictions.macaque_encoding.encoding_things.run
python -m experiments.universality.downstream_predictions.things_behavior.triplets.run
```

These require the recordings and triplets described in the [data layout](../../README.md#data). Run their `dimension_prediction` analyses afterward, followed by [`comparison`](downstream_predictions/comparison/). [`paper_statistics`](paper_statistics/) collects the results once the analyses are complete.
