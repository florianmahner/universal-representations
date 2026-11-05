# Universality

Compute scores from the model embeddings before running downstream analyses:

```bash
python -m experiments.universality.metric.compute.run
```

| Directory | Analysis |
|---|---|
| [metric/validate](metric/validate/) | Null calibration, stability, rank and model-set sensitivity, ObjectNet, CKA, and ridge comparisons |
| [controlled_comparisons](controlled_comparisons/) | Architecture, objective, training data, model size, and accuracy |
| [dimensions](dimensions/) | Category consistency, sparsity, reconstruction importance, and clustering |
| [downstream_predictions/macaque_encoding](downstream_predictions/macaque_encoding/) | Macaque IT encoding |
| [downstream_predictions/things_behavior](downstream_predictions/things_behavior/) | Human odd-one-out judgments |
| [downstream_predictions/comparison](downstream_predictions/comparison/) | Neural and behavioral results across model groups |
| [paper_statistics](paper_statistics/) | Summary statistics for the paper |

Run modules from the repository root, for example:

```bash
python -m experiments.universality.controlled_comparisons.run
```

For downstream predictions, run `macaque_encoding.encoding_things.run` and `things_behavior.triplets.run` before their `dimension_prediction` analyses. Run `paper_statistics.run` last. Neural and behavioral analyses require the corresponding recordings and triplets; see the [data layout](../../README.md#data).
