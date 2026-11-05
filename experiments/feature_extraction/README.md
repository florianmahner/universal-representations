# Feature extraction

Extract penultimate-layer features using the models in [`model_metadata.csv`](../../data/meta/things22k/model_metadata.csv). Install the `.[features]` dependencies first.

```bash
python -m experiments.feature_extraction.extract dataset=things22k
python -m experiments.feature_extraction.extract dataset=objectnet 'models=[torchvision_resnet50_imagenet1k_v2,clip_vit_b_16]'
```

Place images in `<data_root>/images/<dataset>/<category>/<filename>`, following the image metadata CSV. Features are saved to `<data_root>/features/<dataset>/<model>.npy`. Existing files are skipped; failures are recorded in `failed_models.txt`.
