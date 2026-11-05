import contextlib
import gc
import importlib.util
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def validate_source_dependencies(sources: set[str]) -> None:
    """Fail once, before extraction, when a selected source dependency is absent."""
    requirements = {
        "clip": ("clip", "the official OpenAI CLIP package"),
        "open_clip": ("open_clip", "open-clip-torch"),
        "bit_expert": ("tensorflow_hub", "tensorflow and tensorflow-hub"),
        "detectron": ("detectron2", "a Detectron2 build matching your Torch/CUDA stack"),
        "taskonomy": ("visualpriors", "visualpriors"),
    }
    missing = [
        f"{source}: install {description}"
        for source, (module, description) in requirements.items()
        if source in sources and importlib.util.find_spec(module) is None
    ]
    if missing:
        details = "\n  ".join(missing)
        raise RuntimeError(
            "Missing dependencies for selected model sources:\n"
            f"  {details}\nSee README.md#feature-extraction-dependencies."
        )


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


@contextlib.contextmanager
def _temporary_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.pop(0)


def _imagenet_transform():
    from timm.data import create_transform
    return create_transform(input_size=224, is_training=False)


def _pool(x: torch.Tensor, use_cls: bool = False) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x[:, 0] if use_cls else x.mean(dim=1)
    if x.ndim == 4:
        return x.mean(dim=(-2, -1))
    return x.flatten(1)


class _FXWrapper(nn.Module):
    """Wraps a torchvision feature extractor to return the features tensor directly."""

    def __init__(self, extractor: nn.Module):
        super().__init__()
        self.extractor = extractor

    def forward(self, x):
        return self.extractor(x)["features"]


def _create_timm_model(timm_id: str) -> tuple[nn.Module, object]:
    import timm
    from timm.data import resolve_data_config, create_transform

    model = timm.create_model(timm_id, pretrained=True, num_classes=0)
    data_config = resolve_data_config({}, model=model)
    return model, create_transform(**data_config)


def _load_torchvision(load_id: str, penultimate_layer: str, uid: str) -> tuple[nn.Module, object]:
    from torchvision.models import get_model, get_model_weights
    from torchvision.models.feature_extraction import create_feature_extractor

    name = uid.removeprefix("torchvision_")
    variant = name[len(load_id) + 1:].upper()
    weights_enum = get_model_weights(load_id)
    weights = weights_enum[variant]
    model = get_model(load_id, weights=weights)
    extractor = create_feature_extractor(model, return_nodes={penultimate_layer: "features"})
    return _FXWrapper(extractor), weights.transforms()


def _load_clip(load_id: str) -> tuple[nn.Module, object]:
    import clip

    model, transform = clip.load(load_id, device="cpu")
    visual = model.visual.float()
    visual.proj = None  # extract ln_post(CLS) before projection
    return visual, transform


def _load_open_clip(load_id: str, pretrained: str) -> tuple[nn.Module, object]:
    import open_clip

    model, _, transform = open_clip.create_model_and_transforms(load_id, pretrained=pretrained)
    visual = model.visual.float()
    visual.proj = None  # extract ln_post(CLS) before projection
    return visual, transform


def _load_dino(load_id: str) -> tuple[nn.Module, object]:
    model = torch.hub.load("facebookresearch/dino:main", load_id)
    return model, _imagenet_transform()


def _load_ipcl(load_id: str) -> tuple[nn.Module, object]:
    model, transform = torch.hub.load("harvard-visionlab/open_ipcl", load_id)
    model.fc8 = nn.Identity()  # remove projection head, extract fc7 backbone (4096-dim)
    model._l2norm = False
    return model, transform


def _load_vissl(weights_url: str, load_id: str) -> tuple[nn.Module, object]:
    from torchvision.models import resnet50

    model = resnet50()
    model.fc = nn.Identity()
    file_name = f"{load_id}.torch"
    state = torch.hub.load_state_dict_from_url(
        weights_url, map_location="cpu", file_name=file_name
    )

    if "classy_state_dict" in state:
        sd = state["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state

    model_keys = set(model.state_dict().keys())
    converted = {}
    for k, v in sd.items():
        clean = k.replace("_feature_blocks.", "").replace("module.", "")
        if clean in model_keys:
            converted[clean] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
    model.load_state_dict(converted, strict=True)
    return model, _imagenet_transform()


def _load_slip(weights_url: str) -> tuple[nn.Module, object]:
    with _temporary_sys_path(Path(__file__).resolve().parent / "model_code"):
        import slip_models

    ckpt = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
    model_fn = getattr(slip_models, ckpt["args"].model)
    model = model_fn(
        rand_embed=False,
        ssl_mlp_dim=ckpt["args"].ssl_mlp_dim,
        ssl_emb_dim=ckpt["args"].ssl_emb_dim,
    )

    sd = OrderedDict(
        (k.replace("module.", ""), v) for k, v in ckpt["state_dict"].items()
    )
    model.load_state_dict(sd, strict=True)
    return model.visual, _imagenet_transform()


def _load_bit_expert(load_id: str) -> tuple[nn.Module, object]:
    import tensorflow_hub

    with _temporary_sys_path(Path(__file__).resolve().parent / "model_code"):
        from bit_experts import ResNetV2

    url = f"https://tfhub.dev/google/experts/bit/r50x1/in21k/{load_id}/1"
    expert_weights = tensorflow_hub.KerasLayer(url).weights
    weights_dict = {
        w.name.replace(":0", ""): np.array(w) for w in expert_weights
    }

    model = ResNetV2(
        ResNetV2.BLOCK_UNITS["r50"], width_factor=1,
        head_size=1000, zero_head=True, remove_head=True,
    ).load_from(weights_dict)

    return model, _imagenet_transform()


def _load_detectron(load_id: str) -> tuple[nn.Module, object]:
    from detectron2 import model_zoo, modeling
    from detectron2.checkpoint import DetectionCheckpointer

    cfg = model_zoo.get_config(load_id)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(load_id)
    model = modeling.build_model(cfg.clone())
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)

    class _FPNp5(nn.Module):
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


def _load_midas(load_id: str) -> tuple[nn.Module, object]:
    model = torch.hub.load("intel-isl/MiDaS", load_id)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    raw_transform = (
        midas_transforms.dpt_transform if "DPT" in load_id
        else midas_transforms.small_transform
    )

    # MiDaS transforms expect numpy input and add a spurious batch dim
    def transform(img):
        return raw_transform(np.array(img)).squeeze(0)

    return model, transform


def _load_yolov5(load_id: str) -> tuple[nn.Module, object]:
    model = torch.hub.load("ultralytics/yolov5", load_id, pretrained=True, force_reload=False)
    return model, _imagenet_transform()


def _load_taskonomy(load_id: str) -> tuple[nn.Module, object]:
    from torchvision.transforms import functional as F
    from visualpriors.taskonomy_network import (
        TASKONOMY_PRETRAINED_URLS,
        TaskonomyEncoder,
    )

    model = TaskonomyEncoder()
    if load_id != "random_weights":
        weights = torch.utils.model_zoo.load_url(
            TASKONOMY_PRETRAINED_URLS[load_id + "_encoder"]
        )
        model.load_state_dict(weights["state_dict"])

    def transform(img):
        return F.to_tensor(F.resize(img, (256, 256))) * 2 - 1

    return model, transform


def load_model(row: pd.Series) -> tuple[nn.Module, object]:
    source = row["source"]
    load_id = row["load_id"]

    if source in ("timm", "seer"):
        return _create_timm_model(load_id)
    if source == "torchvision":
        return _load_torchvision(load_id, row["penultimate_layer"], row["feature_uid"])
    if source == "clip":
        return _load_clip(load_id)
    if source == "open_clip":
        return _load_open_clip(load_id, row["pretrained"])
    if source == "dino":
        return _load_dino(load_id)
    if source == "ipcl":
        return _load_ipcl(load_id)
    if source == "vissl":
        return _load_vissl(row["weights_url"], load_id)
    if source == "slip":
        return _load_slip(row["weights_url"])
    if source == "bit_expert":
        return _load_bit_expert(load_id)
    if source == "detectron":
        return _load_detectron(load_id)
    if source == "midas":
        return _load_midas(load_id)
    if source == "yolov5":
        return _load_yolov5(load_id)
    if source == "taskonomy":
        return _load_taskonomy(load_id)
    raise ValueError(f"Unknown source: {source}")


def _forward_pool(
    model: nn.Module, loader: DataLoader, device: torch.device,
    use_cls: bool = False,
) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for batch in loader:
            feats.append(_pool(model(batch.to(device)), use_cls=use_cls).cpu())
    return torch.cat(feats).numpy()


def _extract_hooked(
    model: nn.Module, loader: DataLoader, device: torch.device,
    layer_name: str, use_cls: bool = False,
) -> np.ndarray:
    target = model.get_submodule(layer_name)
    captured = []
    handle = target.register_forward_hook(
        lambda _m, _i, o: captured.append(o.detach())
    )
    feats = []
    with torch.no_grad():
        for batch in loader:
            captured.clear()
            model(batch.to(device))
            feats.append(_pool(captured[0], use_cls=use_cls).cpu())
    handle.remove()
    return torch.cat(feats).numpy()


def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device,
    row: pd.Series,
) -> np.ndarray:
    model = model.to(device).eval()
    source = row["source"]

    # Only midas/yolov5/taskonomy need hooked extraction; torchvision handles
    # it internally via FX wrapper, and all other sources use forward+pool.
    penultimate_layer = row.get("penultimate_layer", "")
    hooked = source in {"midas", "yolov5", "taskonomy"}
    if hooked and penultimate_layer and pd.notna(penultimate_layer):
        use_cls = "DPT" in row.get("load_id", "")
        return _extract_hooked(model, loader, device, penultimate_layer, use_cls=use_cls)

    use_cls = source == "dino" and "vit" in row["load_id"]
    return _forward_pool(model, loader, device, use_cls=use_cls)


def extract_model(
    row: pd.Series, image_paths: list[Path],
    device: torch.device, output_dir: Path, batch_size: int = 64,
) -> bool:
    uid = row["feature_uid"]
    output_path = output_dir / f"{uid}.npy"
    if output_path.exists():
        logger.info(f"SKIP {uid}")
        return True

    logger.info(f"START {uid} (source={row['source']}, load_id={row['load_id']})")

    try:
        model, transform = load_model(row)
    except Exception as e:
        logger.error(f"FAIL {uid}: load error: {e}")
        return False

    try:
        # MiDaS transforms emit variable-size tensors, so they cannot be batched.
        bs = 1 if row["source"] == "midas" else batch_size
        loader = DataLoader(
            ImagePathDataset(image_paths, transform),
            batch_size=bs, num_workers=8, pin_memory=True,
        )
        feats = extract_features(model, loader, device, row)
    except Exception as e:
        logger.error(f"FAIL {uid}: extraction error: {e}")
        return False
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if feats.shape[0] != len(image_paths):
        logger.error(f"FAIL {uid}: got {feats.shape[0]} rows, expected {len(image_paths)}")
        return False

    if np.isnan(feats).any() or np.isinf(feats).any():
        logger.error(f"FAIL {uid}: NaN/Inf detected")
        return False

    np.save(output_path, feats.astype(np.float32))
    logger.info(f"OK {uid}: shape={feats.shape}")
    return True


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    output_dir = Path(cfg.data_root) / "features" / cfg.dataset
    meta_dir = Path(cfg.data_root) / "meta" / cfg.dataset
    image_csv = next(meta_dir.glob("image_info*.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    image_rows = pd.read_csv(image_csv)
    required_image_columns = {"path", "category", "filename"}
    missing_columns = required_image_columns.difference(image_rows.columns)
    if missing_columns:
        raise ValueError(
            f"{image_csv} is missing columns: {sorted(missing_columns)}"
        )
    # The catalog's "path" column holds legacy locations; the images are read
    # from the staged tree below data_root instead.
    image_paths = [
        Path(cfg.data_root) / "images" / cfg.dataset / str(row.category) / str(row.filename)
        for row in image_rows.itertuples(index=False)
    ]
    logger.info(f"Images: {len(image_paths)}")

    cols = ["feature_uid", "source", "load_id", "weights_url", "penultimate_layer"]
    meta = pd.read_csv(Path(cfg.data_root) / "meta" / "things22k" / "model_metadata.csv")  # the model set is dataset independent
    if "pretrained" in meta.columns:
        cols.append("pretrained")
    meta = meta[cols]
    if cfg.models:
        meta = meta[meta["feature_uid"].isin(list(cfg.models))]
    logger.info(f"Models: {len(meta)}")
    validate_source_dependencies(set(meta["source"].astype(str)))

    n_ok, n_fail, failed = 0, 0, []
    for _, row in meta.iterrows():
        if extract_model(row, image_paths, device, output_dir):
            n_ok += 1
        else:
            n_fail += 1
            failed.append(row["feature_uid"])

    logger.info(f"DONE: {n_ok} ok, {n_fail} failed")
    if failed:
        fail_path = output_dir / "failed_models.txt"
        fail_path.write_text("\n".join(failed))
        logger.info(f"Failed models written to {fail_path}")


if __name__ == "__main__":
    main()
