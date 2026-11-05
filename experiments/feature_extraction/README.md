### Feature extraction

Extract penultimate-layer representations for the models in [`model_metadata.csv`](../../data/meta/things22k/model_metadata.csv). Install the `.[features]` dependencies first; CLIP and Detectron2 require separate installation.

#### Run

```bash
python -m experiments.feature_extraction.extract dataset=things22k
python -m experiments.feature_extraction.extract dataset=objectnet 'models=[torchvision_resnet50_imagenet1k_v2,clip_vit_b_16]'
```

Run these commands from the repository root. Omit `models` to extract all models in the metadata table.

#### Inputs and outputs

Place images in `<data_root>/images/<dataset>/<category>/<filename>`, following the corresponding image metadata CSV. Each output is an image-by-feature array at `<data_root>/features/<dataset>/<model>.npy`. Existing files are skipped; failures are recorded in `failed_models.txt`.
