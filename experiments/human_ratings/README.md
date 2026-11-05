### Human dimension ratings

Create rating tasks and analyze judgments of whether dimensions reflect semantic properties, visual properties, both, or neither.

#### Run

```bash
python -m experiments.human_ratings.generate_tasks
python -m experiments.human_ratings.parse_assignments
python -m experiments.human_ratings.run
```

Run from the repository root in this order. Task generation requires the dimension clustering and universality scores. Parsing requires the participant exports at `data/ratings/assignments_*.csv`.

#### Included files

Participant-level exports are not distributed. The aggregated labels are in [`dimension_ratings.csv`](../../data/ratings/dimension_ratings.csv). [`template.html`](template.html) contains the study interface, with placeholder consent links.
