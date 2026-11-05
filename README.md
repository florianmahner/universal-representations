# Universal representations

Code for **Characterizing Universal Object Representations Across Vision Models**.
The pipeline factorizes model similarity matrices
into non-negative dimensions and scores how universal each dimension is across
models:

1. [`experiments/feature_extraction`](experiments/feature_extraction/) — features for the 22,248 THINGS images
2. [`experiments/factorization`](experiments/factorization/) — symmetric NMF over a bandwidth x seed grid, then bandwidth and seed selection
3. [`experiments/universality`](experiments/universality/) — dimension-level universality and everything downstream of it
4. [`experiments/rebuttal`](experiments/rebuttal/) — controls and follow-up analyses
5. [`experiments/human_ratings`](experiments/human_ratings/) — the dimension rating study

Each of those folders has a README listing its analyses and what each one
reproduces.

## Setup

Python 3.10 or newer:

```bash
git clone https://github.com/florianmahner/universal-representations.git
cd universal-representations
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

The tests use synthetic data and need none of the large inputs. Optional
extras: `.[features]` for the torch/timm/OpenCLIP/tensorflow extraction stack
(`clip` and `detectron2` need their own upstream install), `.[analysis]` for
the semantic-label and AligNet steps, `.[reports]` for the cluster content
analysis.

`snmf/` is the bundled version of [pysrf](https://github.com/florianmahner/pysrf),
renamed for the anonymous release. It retains the implementation used for these
analyses rather than following later upstream changes.
`snmf/_bsum.pyx` is the Cython source of the same inner solver, included for
reference; building it needs a Cython toolchain that this package does not
configure. The full 22k x 22k fits were cluster jobs, not laptop-scale.

## Running

Every analysis is a module, run from the repository root:

```bash
python -m experiments.factorization.fit.run_grid models=[resnet50]
python -m experiments.universality.metric.compute.run
```

Results are written to an `outputs/` folder beside the module that produced
them.

## Configuration

`config.yaml` is the entire configuration: where the data lives and the few
settings that vary between runs. Any value can be overridden on the command
line as `key=value`:

```bash
python -m experiments.universality.metric.compute.run data_root=/scratch/things n_jobs=8
```

The `factorization` block pins the published protocol (rank 50, the eight
bandwidth multipliers, five seeds). Changing it changes what the numbers mean.
Permutation counts and the other analysis constants live in the code that uses
them.

## Data

Point `data_root` in `config.yaml` at a directory with this layout:

```text
data/
├── images/things/<category>/<image>
├── features/{things22k,objectnet}/<model>.npy
├── features/macaque/22k/processed/{F,N}.npy
├── embeddings/<dataset>/<model>/rank_<k>/sigma_<m>/seed_<s>/{w.npz,metrics.json}
├── behavior/things_triplets/
├── ratings/          (bundled)
└── meta/             (bundled)
```

Feature files are 2-D arrays, one row per image, in the order given by
`data/meta/<dataset>/image_info*.csv`; the THINGS matrices have 22,248 rows.

Bundled: model and image metadata, bandwidth-selection tables, THINGS concept
metadata, and the aggregated human dimension ratings (570 participants, 432
after attention checks).

Not bundled: THINGS and ObjectNet images, model weights and activations,
behavioral triplets, macaque recordings, participant-level rating exports, the
IPCL checkpoints, and the rating-study image grids.

`python -m experiments.universality.metric.compute.run` writes the
per-dimension universality tables that most downstream analyses read, so it
has to run before them.

## License

Project code is released under the [BSD 3-Clause License](LICENSE).
The bundled optimizer retains its [original license](snmf/LICENSE);
see [third-party notices](THIRD_PARTY.md) for other included code and data.
