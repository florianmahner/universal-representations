# Human dimension ratings

Crowd-sourced labeling of NMF dimensions as *semantic*, *visual*, *both*, or *neither* (Appendix, dimension rating study). Representative dimensions are picked by clustering, rated online, and the labels related to per-dimension universality. `template.html` contains the task interface; institutional consent links have been replaced by placeholders.

| Script | Reproduces | Run |
|---|---|---|
| [`generate_tasks.py`](generate_tasks.py) | Study design: cluster representatives split into survey versions with catch, attention-check, and repeat trials | `python -m experiments.human_ratings.generate_tasks` |
| [`parse_assignments.py`](parse_assignments.py) | Per-trial tables from the raw participant exports | `python -m experiments.human_ratings.parse_assignments` |
| [`run.py`](run.py) | Fig. 2b and the reliability and confusion figures in the appendix | `python -m experiments.human_ratings.run` |

Run them in this order:

```bash
python -m experiments.human_ratings.generate_tasks    # writes outputs/tasks.csv
python -m experiments.human_ratings.parse_assignments # needs data/ratings/assignments_*.csv
python -m experiments.human_ratings.run               # QC, reliability, Fig. 2b
```

Requires from `experiments/universality/`: the clustering linkage (`dimensions/clustering/outputs/linkage.npy`) and universality scores (`shared/data/universality_things22k_rank_50.csv`). The raw participant exports (`data/ratings/assignments_*.csv`) are not shipped; the anonymized aggregate they produce is included at `data/ratings/dimension_ratings.csv`.
