# Follow-up analyses

Controls and extensions of the main result. Each analysis reruns the published pipeline (RBF kernel, rank-50 symmetric NMF over the bandwidth x seed grid, `cos2_perm_adj` universality against the 162-model reference) with only the input representation or the matching rule changed: other network depths, untrained networks, pixels, noise, further model families, plus label grounding without human judgments. Upstream are `experiments/factorization` and `experiments/universality`, whose helpers every module here imports.

| Analysis | Reproduces | Run |
|---|---|---|
| [`universality_across_layers/`](universality_across_layers/) | Universality at normalized depths 0.1 / 0.5 / 0.9 plus untrained penultimate layers for all architectures | `python -m experiments.rebuttal.universality_across_layers.scoring.score` (chain below) |
| [`universality_across_layers/human_ratings/`](universality_across_layers/human_ratings/) | Layer-wise repeat of the dimension rating study on early/middle/late dimensions | `python -m experiments.rebuttal.universality_across_layers.human_ratings.{cluster_layers, render_grids, generate_tasks, parse_assignments, plot_pilot}` |
| [`baselines/pixel_models/`](baselines/pixel_models/) | Image-only baselines: raw pixels, their first principal components, Gabor energy | `python -m experiments.rebuttal.baselines.pixel_models.{build_features, factorize, select_sigma, score}` |
| [`baselines/random_features/`](baselines/random_features/) | Gaussian noise representations that never saw an image: the noise floor of the measure | `python -m experiments.rebuttal.baselines.random_features.run` |
| [`baselines/summary/`](baselines/summary/) | The control ladder in one table and figure: trained models, untrained networks, pixel PCA, random features | `python -m experiments.rebuttal.baselines.summary.{collect, plot_essentials}` |
| [`ipcl/`](ipcl/) | Training-data domain with architecture and objective fixed (AlexNet-GN on four datasets) | `python -m experiments.rebuttal.ipcl.run` |
| [`alignnet/`](alignnet/) | Universality of AligNet models post-trained toward human similarity, against their base backbones | `python -m experiments.rebuttal.alignnet.{run, visualize_vit_b}` |
| [`greedy_matching/`](greedy_matching/) | Greedy best-match vs Hungarian assignment (Appendix, greedy matching) | `python -m experiments.rebuttal.greedy_matching.{run, plot}` |
| [`categorical_control/`](categorical_control/) | Category consistency partialled out of the predictivity results (Appendix, category consistency) | `python -m experiments.rebuttal.categorical_control.run` |
| [`label_grounding/`](label_grounding/) | Ridge prediction of every dimension from semantic (BERT, WordNet) and visual (image statistics, untrained conv1) references | `python -m experiments.rebuttal.label_grounding.{extract_torch, run, plot}` |
| [`llm_dimension_descriptions/`](llm_dimension_descriptions/) | Open-ended descriptions of the representative dimensions by vision-language models, coded and compared to the human labels | `python -m experiments.rebuttal.llm_dimension_descriptions.{prepare_panels, run, additional_*, plot}` |
| [`predictions_across_depth/`](predictions_across_depth/) | Universality vs THINGS triplet accuracy and macaque IT encoding at each depth | `python -m experiments.rebuttal.predictions_across_depth.{triplets, macaque, plots}` |

Braces list the stages of one analysis, in order. Only the depth sweep has an order across folders:

```bash
python -m experiments.rebuttal.universality_across_layers.layer_selection.make_spec        # which block sits at depth 0.1/0.5/0.9 per model
python -m experiments.rebuttal.universality_across_layers.feature_extraction.extract_full  # depth features; untrained_pen for the untrained set
python -m experiments.rebuttal.universality_across_layers.factorization.factorize          # then select_sigma, artifacts
python -m experiments.rebuttal.universality_across_layers.scoring.score                    # then figures.plots
```

Not shipped: the THINGS images under `data/images/things/`, the IPCL and AligNet checkpoints (downloaded by their own modules), and the depth features these modules write. Scoring anything against the reference needs the rank-50 embeddings from `experiments/factorization/fit` and the universality tables from `experiments/universality/metric/compute`.
