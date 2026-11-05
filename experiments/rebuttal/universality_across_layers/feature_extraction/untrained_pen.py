from __future__ import annotations

import gc
import logging
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from experiments.feature_extraction.extract import ImagePathDataset, _imagenet_transform

logger = logging.getLogger(__name__)


def slug(arch: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", arch).strip("_").lower()


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def build_untrained(row: pd.Series):
    """Return (model, transform) with random init; model(x) -> penultimate (B, d)."""
    source, load_id = row["source"], row["load_id"]
    _seed()
    if source in ("timm", "seer", "dino"):
        import timm
        from timm.data import resolve_data_config, create_transform
        if source == "dino":  # dino_vitb16 is a plain ViT-B/16 under the hood
            load_id = {"dino_vitb16": "vit_base_patch16_224"}.get(load_id, load_id)
        model = timm.create_model(load_id, pretrained=False, num_classes=0)
        data_cfg = resolve_data_config({}, model=model)
        return model, create_transform(**data_cfg)
    if source == "torchvision":
        from torchvision.models import get_model
        from torchvision.models.feature_extraction import create_feature_extractor
        model = get_model(load_id, weights=None)
        extractor = create_feature_extractor(
            model, return_nodes={row["penultimate_layer"]: "features"})

        class _W(torch.nn.Module):
            def __init__(self, e):
                super().__init__()
                self.e = e

            def forward(self, x):
                return self.e(x)["features"]
        return _W(extractor), _imagenet_transform()
    if source == "clip":
        import open_clip
        oc_name = load_id.replace("/", "-")
        model, _, transform = open_clip.create_model_and_transforms(oc_name, pretrained=None)
        _seed()  # re-seed: creation order differs from clip pkg; fix init determinism
        visual = model.visual.float()
        visual.proj = None
        return visual, transform
    if source == "bit_expert":
        import sys
        sys.path.insert(0, str(Path("experiments/feature_extraction/model_code").resolve()))
        from bit_experts import ResNetV2
        model = ResNetV2(ResNetV2.BLOCK_UNITS["r50"], width_factor=1,
                         head_size=1000, zero_head=True, remove_head=True)
        return model, _imagenet_transform()
    if source == "slip":
        import sys
        sys.path.insert(0, str(Path("experiments/feature_extraction/model_code").resolve()))
        import slip_models
        # arch strings ViT-S / ViT-B / ViT-L -> SLIP constructors, random init
        key = {"ViT-S": "SLIP_VITS16", "ViT-B": "SLIP_VITB16", "ViT-L": "SLIP_VITL16"}[
            row["architecture"]]
        model = getattr(slip_models, key)(rand_embed=False, ssl_mlp_dim=4096,
                                          ssl_emb_dim=256)
        return model.visual, _imagenet_transform()
    if source == "detectron":
        from detectron2 import model_zoo, modeling
        zoo_cfg = model_zoo.get_config(load_id)
        model = modeling.build_model(zoo_cfg.clone())  # random init: no checkpoint load
        model.eval()
        pixel_mean = torch.tensor(zoo_cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        pixel_std = torch.tensor(zoo_cfg.MODEL.PIXEL_STD).view(3, 1, 1)

        class _FPNp5(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                return self.backbone(x)["p5"]

        def transform(img):
            arr = np.asarray(img.resize((256, 256)))[:, :, ::-1]
            t = torch.as_tensor(arr.astype("float32").transpose(2, 0, 1))
            return (t - pixel_mean) / pixel_std
        return _FPNp5(model.backbone), transform
    raise RuntimeError(f"no untrained builder for source={source}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv(Path(cfg.data_root) / "meta" / "things22k" / "model_metadata.csv")
    # representative per unique architecture, preferring buildable sources
    prio = {"timm": 0, "seer": 1, "torchvision": 2, "clip": 3, "bit_expert": 4,
            "dino": 5, "open_clip": 6, "slip": 7}
    meta["prio"] = meta.source.map(prio).fillna(99)
    reps = meta.sort_values("prio").drop_duplicates("architecture")
    reps = reps[reps.source != "taskonomy"]  # taskonomy_random_weights is a cohort model

    info = pd.read_csv(Path(cfg.data_root) / "meta" / "things22k" / "image_info_22k.csv")
    images_root = Path(cfg.data_root) / "images" / "things"
    paths = [str(images_root / c / f) for c, f in zip(info["category"], info["filename"])]

    todo = list(reps.iterrows())
    logger.info(f"{len(todo)} untrained archs")

    # Shared with the layer-wise extractions: the factorization stage globs this
    # directory for every representation it fits.
    # Same directory extract_full.py writes to; the factorization stage globs it.
    feature_dir = Path(__file__).parent / "outputs" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    skips = []
    for _, row in todo:
        out_path = feature_dir / f"untrained__{slug(row['architecture'])}__pen.npy"
        if out_path.exists():
            continue
        try:
            model, transform = build_untrained(row)
            model.eval().to(device)
            dl = DataLoader(ImagePathDataset(paths, transform), batch_size=64,
                            num_workers=10, shuffle=False, pin_memory=True)
            bufs = []
            with torch.no_grad():
                for xb in dl:
                    o = model(xb.to(device, non_blocking=True))
                    if o.ndim > 2:
                        o = o.flatten(1) if o.ndim == 2 else o.mean(dim=tuple(range(2, o.ndim)))
                    bufs.append(o.float().cpu())
            feats = torch.cat(bufs).numpy().astype(np.float32)
            assert feats.shape[0] == len(paths) and np.isfinite(feats).all()
            np.save(out_path, feats)
            logger.info(f"{row['architecture']} ({row['source']}/{row['load_id']}): {feats.shape}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"SKIP {row['architecture']} ({row['source']}): {e}")
            skips.append(dict(architecture=row["architecture"], source=row["source"],
                              error=f"{type(e).__name__}: {e}",
                              trace=traceback.format_exc()[-300:]))
    if skips:
        out = Path(__file__).parent / "outputs"
        out.mkdir(exist_ok=True)
        skip_csv = out / "untrained_skip_report.csv"
        prev = pd.read_csv(skip_csv) if skip_csv.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(skips)]).to_csv(skip_csv, index=False)
    logger.info(f"done, {len(skips)} skips")


if __name__ == "__main__":
    main()
