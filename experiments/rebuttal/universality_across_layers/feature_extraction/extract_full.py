from __future__ import annotations

import gc
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from experiments.feature_extraction.extract import ImagePathDataset
from experiments.rebuttal.universality_across_layers.layer_selection.make_spec import load_for_layers

logger = logging.getLogger(__name__)

DEPTH_TAG = {0.1: "d010", 0.5: "d050", 0.9: "d090"}


def image_paths(data_root: Path) -> list[str]:
    info = pd.read_csv(data_root / "meta" / "things22k" / "image_info_22k.csv")
    paths = [
        str(data_root / "images" / "things" / c / f)
        for c, f in zip(info["category"], info["filename"])
    ]
    missing = [p for p in paths[:50] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"image sample missing, e.g. {missing[0]}")
    return paths


def _pool_out(out: torch.Tensor, swin: bool, use_cls: bool, bs: int) -> torch.Tensor:
    if out.ndim == 4:
        if swin:
            return out.mean(dim=(1, 2))
        # 4D token layouts (e.g. NesT: (B, n_blocks, tokens, C)) are channels-last
        # and non-square in dims 2/3; conv maps (B, C, H, W) are square (224 crops).
        if out.shape[2] != out.shape[3]:
            return out.mean(dim=(1, 2))
        return out.mean(dim=(2, 3))
    if out.ndim == 3:
        # clip/open_clip transformers run sequence-first (L, B, C); detect by
        # locating the batch dimension and normalize to (B, L, C).
        if out.shape[0] != bs and out.shape[1] == bs:
            out = out.permute(1, 0, 2)
        return out[:, 0] if use_cls else out.mean(dim=1)
    return out.flatten(1)


def extract_model(row_meta: pd.Series, spec: pd.DataFrame, paths: list[str],
                  device: torch.device, feature_dir: Path,
                  batch_size: int = 64, workers: int = 10) -> None:
    uid = row_meta["feature_uid"]
    out_paths = {r["target_depth"]: feature_dir / f"{uid}__{DEPTH_TAG[r['target_depth']]}.npy"
                 for _, r in spec.iterrows()}
    if all(p.exists() for p in out_paths.values()):
        logger.info(f"{uid}: exists, skip")
        return

    model, transform = load_for_layers(row_meta)
    model.eval().to(device)
    mods = dict(model.named_modules())
    use_cls = row_meta["source"] in {"clip", "open_clip", "slip"}

    buffers: dict[float, list] = {t: [] for t in out_paths}
    cur_bs = [0]  # set per batch before forward; hooks read it for layout checks
    handles = []
    for _, r in spec.iterrows():
        mod = mods[r["module"]]
        swin = "Swin" in r["trunk_class"]

        def mk(t=r["target_depth"], swin=swin):
            def hook(_m, _i, out):
                if isinstance(out, (tuple, list)):
                    tensors = [o for o in out if torch.is_tensor(o)]
                    out = tensors[-1] if tensors else None
                if torch.is_tensor(out):
                    buffers[t].append(_pool_out(out.detach(), swin, use_cls,
                                                cur_bs[0]).float().cpu())
            return hook
        handles.append(mod.register_forward_hook(mk()))

    ds = ImagePathDataset(paths, transform)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                    shuffle=False, pin_memory=True)
    try:
        with torch.no_grad():
            for xb in dl:
                cur_bs[0] = xb.shape[0]
                model(xb.to(device, non_blocking=True))
    finally:
        for h in handles:
            h.remove()

    for t, buf in buffers.items():
        feats = torch.cat(buf).numpy().astype(np.float32)
        assert feats.shape[0] == len(paths), f"{uid} d={t}: {feats.shape}"
        assert np.isfinite(feats).all(), f"{uid} d={t}: non-finite values"
        np.save(out_paths[t], feats)
        logger.info(f"{uid} d={t}: {feats.shape} -> {out_paths[t].name}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    meta = pd.read_csv(
        Path(cfg.data_root) / "meta" / "things22k" / "model_metadata.csv"
    ).set_index("feature_uid", drop=False)
    spec_all = pd.read_csv(
        Path("experiments/rebuttal/universality_across_layers/spec/layer_spec_full.csv")
    )
    uids = sorted(spec_all["feature_uid"].unique())
    if cfg.models:
        uids = [u for u in uids if u in cfg.models]
    paths = image_paths(Path(cfg.data_root))
    logger.info(f"{len(uids)} models")

    skips = []
    for uid in uids:
        try:
            extract_model(meta.loc[uid], spec_all[spec_all["feature_uid"] == uid],
                          paths, device, feature_dir)
        except Exception as e:
            logger.warning(f"SKIP {uid}: {e}")
            skips.append(dict(feature_uid=uid, error=f"{type(e).__name__}: {e}",
                              trace=traceback.format_exc()[-400:]))
            gc.collect()
            torch.cuda.empty_cache()
    if skips:
        skip_csv = out / "extract_skip_report.csv"
        prev = pd.read_csv(skip_csv) if skip_csv.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(skips)]).to_csv(skip_csv, index=False)
    logger.info(f"done, {len(skips)} skips")


if __name__ == "__main__":
    main()
