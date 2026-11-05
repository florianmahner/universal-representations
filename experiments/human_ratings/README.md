# Human dimension ratings

Generate rating tasks, parse participant exports, and analyze semantic/visual labels:

```bash
python -m experiments.human_ratings.generate_tasks
python -m experiments.human_ratings.parse_assignments
python -m experiments.human_ratings.run
```

Requires the dimension clustering and universality scores. Participant exports (`data/ratings/assignments_*.csv`) are not distributed; aggregated labels are in [`dimension_ratings.csv`](../../data/ratings/dimension_ratings.csv).

[`template.html`](template.html) contains the study interface with placeholder consent links.
