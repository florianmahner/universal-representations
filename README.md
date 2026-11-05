<img src="assets/logo.png" alt="Universal representations" width="130" align="right">

# Characterizing universal object representations across vision models

Code for our [NeurIPS 2026 paper](https://arxiv.org/abs/2605.13675). We extract non-negative dimensions from 162 vision models and measure how consistently they recur across models.

### Citation

Please cite our paper if you use this code. Citation metadata are in [CITATION.cff](CITATION.cff), also available through **Cite this repository** on GitHub.

### Getting non-negative embeddings

The factorization is provided by **[pysrf](https://github.com/florianmahner/pysrf)**, our library for **Similarity-based representation factorization**. Use it directly to extract nonnegative dimensions from your own similarity matrices:

```python
from pysrf import SRF

weights = SRF(rank=50, random_state=42).fit_transform(similarity)
```

Here, `similarity` is a symmetric, nonnegative similarity matrix. The fitting settings used in this paper are in [`config.yaml`](config.yaml).

### Installation

Python 3.10+ and a C compiler are required. Installation includes `pysrf` from a fixed GitHub commit.

```bash
git clone https://github.com/florianmahner/universal-representations.git
cd universal-representations
make install
source .venv/bin/activate
```

`make install` creates `.venv` and installs the dependencies. Run `make check` to test the installation with synthetic data.

For feature extraction, use `make install EXTRAS=test,features`. Other extras are `analysis` (semantic-label and AligNet analyses) and `reports` (cluster reports). CLIP and Detectron2 require separate installation.

### Usage

Fit embeddings for a model from its extracted features:

```bash
python -m experiments.factorization.fit.run_grid 'models=[torchvision_resnet50_imagenet1k_v2]'
```

Once embeddings are available for the model cohort, compute universality scores:

```bash
python -m experiments.universality.metric.compute.run data_root=/path/to/data
```

Instructions for each analysis:

- [Feature extraction](experiments/feature_extraction/)
- [SRF and bandwidth selection](experiments/factorization/)
- [Universality and downstream predictions](experiments/universality/)
- [Controls and additional analyses](experiments/rebuttal/)
- [Human dimension ratings](experiments/human_ratings/)

### Data

The [Zenodo release](https://zenodo.org/records/23979573) contains the selected rank-50 embeddings for all 162 models: **THINGS (1.05 GB)** and **ObjectNet (2.33 GB)**, with image-order metadata and the selected bandwidths and seeds.

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). DOI: [10.5281/zenodo.23979573](https://doi.org/10.5281/zenodo.23979573).

Download either or both archives and extract them from the repository root:

```bash
curl -fL https://zenodo.org/records/23979573/files/things22k-rank50.zip -o things22k-rank50.zip
unzip -n things22k-rank50.zip

curl -fL https://zenodo.org/records/23979573/files/objectnet-rank50.zip -o objectnet-rank50.zip
unzip -n objectnet-rank50.zip
```

The archives create `data/embeddings/<dataset>/` and include the corresponding metadata under `data/meta/<dataset>/`. Each `w.npz` contains an image-by-50 array named `w`; its rows follow the image metadata CSV. The original precision is preserved.

Aggregated human ratings are included in [`dimension_ratings.csv`](data/ratings/dimension_ratings.csv). Images, extracted features, neural recordings, and behavioral triplets are separate inputs required only for the analyses that use them. Set `data_root` in [`config.yaml`](config.yaml) if you store data elsewhere.

### License

[BSD-3-Clause](LICENSE). See [third-party notices](THIRD_PARTY.md).
