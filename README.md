# Universal representations

Official code for [Characterizing Universal Object Representations Across Vision Models](https://openreview.net/forum?id=16SGONh3Vl), accepted at **NeurIPS 2026**.

## Setup

Python 3.10+.

```bash
git clone https://github.com/florianmahner/universal-representations.git
cd universal-representations
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

Optional dependencies: `.[features]` for feature extraction, `.[analysis]` for semantic-label and AligNet analyses, and `.[reports]` for cluster reports. CLIP and Detectron2 require separate installation.

## Usage

Set paths in [`config.yaml`](config.yaml), or override them on the command line:

```bash
python -m experiments.factorization.fit.run_grid models=[resnet50]
python -m experiments.universality.metric.compute.run data_root=/path/to/data
```

Analysis instructions:

- [Feature extraction](experiments/feature_extraction/)
- [Factorization](experiments/factorization/)
- [Universality](experiments/universality/)
- [Additional analyses](experiments/rebuttal/)
- [Human ratings](experiments/human_ratings/)

## Data

Metadata and aggregated human ratings are included. Images, model weights, features, embeddings, neural recordings, behavioral triplets, and participant-level ratings are not included.

Expected layout under `data_root`:

```text
images/things/<category>/<image>
features/{things22k,objectnet}/<model>.npy
features/macaque/22k/processed/{F,N}.npy
embeddings/<dataset>/<model>/rank_<k>/sigma_<m>/seed_<s>/{w.npz,metrics.json}
behavior/things_triplets/
ratings/
meta/
```

Feature rows must follow `data/meta/<dataset>/image_info*.csv`. Compute universality scores before running downstream analyses. Tests use synthetic data.

## License

[BSD-3-Clause](LICENSE). The bundled optimizer in `snmf/` is derived from [pysrf](https://github.com/florianmahner/pysrf). See [third-party notices](THIRD_PARTY.md).
