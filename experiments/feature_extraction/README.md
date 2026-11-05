# Feature extraction

First stage of the pipeline. Extracts penultimate-layer representations from every model for every image of an image set and writes one `float32` array of shape (n_images, d) per model.

| Analysis | Reproduces | Run |
|---|---|---|
| [`extract.py`](extract.py) | Sec. 3.1: penultimate features over the THINGS images; run on ObjectNet for the image-set control | `python -m experiments.feature_extraction.extract` |
| [`model_code/`](model_code/) | Architecture definitions for the SLIP and BiT-expert checkpoints; imported by `extract.py`, not run directly | — |

Models are defined by `data/meta/things22k/model_metadata.csv` (`source`, `load_id`, `weights_url`, `penultimate_layer`). Extraction is resumable: existing `.npy` files are skipped and failures land in `failed_models.txt` instead of aborting.

```bash
python -m experiments.feature_extraction.extract dataset=things22k
python -m experiments.feature_extraction.extract dataset=objectnet models=[resnet50,clip_vit_b_16]
```

Features go to `<data_root>/features/<dataset>/<feature_uid>.npy`, which `experiments/factorization/fit` reads. Not shipped: the images (`<data_root>/images/<dataset>/<category>/<filename>`, listed in `data/meta/<dataset>/image_info*.csv`) and model weights, downloaded on first use. The stack is the `.[features]` extra; `clip` and `detectron2` install from upstream, the BiT experts need TensorFlow Hub.
