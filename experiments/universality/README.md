# Universality

Everything downstream of the factorization: the universality score itself, its validation, what
separates universal from model specific dimensions, whether inductive biases explain universality,
and how universality relates to macaque IT activity and human similarity judgments. All analyses
read the rank 50 embeddings written by `experiments/factorization` and score every dimension
against the full 162 model set.

| Analysis | Reproduces | Run |
|---|---|---|
| [`metric/compute/`](metric/compute/) | Per-dimension universality across ranks; the scores every other analysis reads (Sec. 4) | `python -m experiments.universality.metric.compute.run` |
| [`metric/validate/null_correction/`](metric/validate/null_correction/), [`null_fraction/`](metric/validate/null_fraction/) | Metric calibration: how often source dimensions go unselected, and the null adjustment that removes the chance floor | `...metric.validate.null_fraction.run`, `...metric.validate.null_correction.run` |
| [`metric/validate/within_model_stability/`](metric/validate/within_model_stability/) | Within-model stability ceiling of the metric | `...metric.validate.within_model_stability.run` |
| [`metric/validate/dataset_comparison/`](metric/validate/dataset_comparison/), [`proxy_validation/`](metric/validate/proxy_validation/), [`cka_comparison/`](metric/validate/cka_comparison/) | Metric validation figure: ObjectNet image-set control, ridge-R² proxy, CKA comparison | `...metric.validate.proxy_validation.run`, `...cka_comparison.run`, then the `plot` modules |
| [`metric/validate/model_set_sensitivity/`](metric/validate/model_set_sensitivity/) | Stability of the scores under model subsampling and leave-family-out | `...metric.validate.model_set_sensitivity.run` |
| [`metric/validate/rank_consistency/`](metric/validate/rank_consistency/) | Rank-50 model rankings against other factorization ranks | `...metric.validate.rank_consistency.plot` |
| [`metric/validate/metric_comparison/`](metric/validate/metric_comparison/) | Greedy vs permutation matching and their collision rates | `...metric.validate.metric_comparison.run` |
| [`controlled_comparisons/`](controlled_comparisons/) | Fig. 3 and Sec. 4.2: all ten contrasts, plus the accuracy and model-size confounds | `python -m experiments.universality.controlled_comparisons.run` |
| [`dimensions/category_consistency/`](dimensions/category_consistency/) | Fig. 2c,d: category consistency vs universality | `...dimensions.category_consistency.run` |
| [`dimensions/reconstruction_importance/`](dimensions/reconstruction_importance/) | Fig. 2e: universality vs a dimension's reconstruction importance | `...dimensions.reconstruction_importance.run` |
| [`dimensions/sparsity/`](dimensions/sparsity/) | Universal dimensions are not a model's strongest ones | `...dimensions.sparsity.run` |
| [`dimensions/characterization/`](dimensions/characterization/) | Sparsity, text-embedding predictability and within-model weight share against universality | `...dimensions.characterization.run` |
| [`dimensions/clustering/`](dimensions/clustering/) | Clustering of all dimensions into the representatives rated in the human study | `...dimensions.clustering.run` |
| [`clustering/characterization/`](dimensions/clustering/characterization/), [`content_analysis/`](dimensions/clustering/content_analysis/), [`rank_stability/`](dimensions/clustering/rank_stability/), [`sposedims/`](dimensions/clustering/sposedims/) | Cluster content, cluster stability across ranks, overlap with SPoSE dimensions | `...dimensions.clustering.<name>.run` |
| [`dimensions/idiosyncratic/`](dimensions/idiosyncratic/) | Taxonomy of low-universality dimensions and which model groups share them | `...dimensions.idiosyncratic.run` |
| [`dimensions/specificity/`](dimensions/specificity/) | Within-group vs between-group dimension matching, with permutation tests and bootstrap CIs | `...dimensions.specificity.run` |
| [`downstream_predictions/macaque_encoding/`](downstream_predictions/macaque_encoding/) | Fig. 4a: model universality vs macaque IT encoding, universal vs specific dimension halves | `...macaque_encoding.encoding_things.run`, then `.dimension_prediction.run` and `.confounds.run` |
| [`downstream_predictions/things_behavior/`](downstream_predictions/things_behavior/) | Fig. 4b: universality vs THINGS odd-one-out triplet accuracy | `...things_behavior.triplets.run`, then `.dimension_prediction.run` |
| [`downstream_predictions/comparison/`](downstream_predictions/comparison/) | Neural and behavioral predictivity side by side across model groups | `...downstream_predictions.comparison.run` |
| [`paper_statistics/`](paper_statistics/) | Collects the correlations reported in Sec. 4.3 from the outputs above into one table | `python -m experiments.universality.paper_statistics.run` |
| [`shared/`](shared/) | Matching metrics, permutation null and null adjustment used by every row above | imported, not run |

Run them in this order: `metric/compute` first, since every other module reads its scores; inside
`downstream_predictions`, `encoding_things` and `triplets` before the `dimension_prediction`,
`confounds` and `comparison` modules that read their CSVs; `paper_statistics` last.

Needs inputs the release does not ship: the embeddings from `experiments/factorization/fit`, THINGS
and ObjectNet features from `experiments/feature_extraction`, macaque IT recordings under
`data/features/macaque/`, THINGS triplets under `data/behavior/`, and, for
`dimensions/category_consistency/relate_to_ratings.py`, the aggregated labels collected by
[`experiments/human_ratings`](../human_ratings/).
