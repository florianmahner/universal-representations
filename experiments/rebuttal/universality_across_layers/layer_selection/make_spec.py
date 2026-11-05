from __future__ import annotations

import gc
import logging
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from experiments.feature_extraction.extract import load_model

logger = logging.getLogger(__name__)

TARGETS = [0.1, 0.5, 0.9]
MIN_BLOCKS = 3


def load_for_layers(row) -> tuple:
    """Like extract.load_model, but torchvision models are returned RAW.

    The canonical loader wraps torchvision models in an FX feature-extractor
    whose traced graph never invokes the block modules' forward(), so hooks on
    Bottleneck/DenseLayer/etc. would not fire. For layer work we hook inner
    modules and ignore the head, so the raw module tree is what we need.
    """
    if row["source"] == "torchvision":
        from torchvision.models import get_model, get_model_weights
        name = row["feature_uid"].removeprefix("torchvision_")
        load_id = row["load_id"]
        variant = name[len(load_id) + 1:].upper()
        weights = get_model_weights(load_id)[variant]
        return get_model(load_id, weights=weights), weights.transforms()
    return load_model(row)


def _dummy_input(transform) -> torch.Tensor:
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    x = transform(img)
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x))
    return x.unsqueeze(0)


def trace_blocks(model: torch.nn.Module, x: torch.Tensor) -> list[dict]:
    """Run one forward, return trunk-block records in execution-completion order."""
    name_of = {m: n for n, m in model.named_modules()}
    events = []  # (module, name, class, out_shape) in completion order
    handles = []

    def hook(mod, inp, out):
        # TNT-style blocks return (pixel_embed, patch_embed) tuples; take the
        # last tensor element (the semantic stream used downstream).
        if isinstance(out, (tuple, list)):
            tensors = [o for o in out if torch.is_tensor(o)]
            out = tensors[-1] if tensors else None
        if torch.is_tensor(out):
            events.append((name_of[mod], type(mod).__name__, tuple(out.shape)))

    for m in name_of:
        if name_of[m]:  # skip the root
            handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    by_class = defaultdict(list)
    seen_names = set()
    for name, cls, shape in events:
        # a module reused at several points in the graph (e.g. CoaT's shared
        # ConvPosEnc) fires multiple times but is ONE hookable module — count
        # it once, at its first firing.
        if name in seen_names:
            continue
        seen_names.add(name)
        by_class[cls].append((name, shape))
    # trunk = the non-trivial class with the most executed instances; ties break
    # toward the class whose instances are deepest in the name hierarchy (real
    # blocks, not e.g. LayerNorm which would win by count) — so exclude classes
    # that are leaf primitives.
    primitive = {
        "Linear", "Conv2d", "LayerNorm", "BatchNorm2d", "GELU", "ReLU", "SiLU",
        "Dropout", "Identity", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
        "Sigmoid", "Hardswish", "Hardsigmoid", "Flatten", "GroupNorm", "Mlp",
        "LayerScale", "DropPath", "Attention", "Softmax", "QuickGELU",
        "Sequential", "ModuleList", "Conv1d", "BatchNorm1d", "Tanh",
    }
    candidates = {c: v for c, v in by_class.items()
                  if len(v) >= MIN_BLOCKS and c not in primitive}
    if not candidates:
        # blockless nets (AlexNet/VGG-style): fall back to the conv layers with
        # feature-map (4D) outputs as the trunk units.
        convs = [(n, s) for n, cls, s in events if cls == "Conv2d" and len(s) == 4]
        if len(convs) >= MIN_BLOCKS:
            candidates = {"Conv2d": convs}
        else:
            raise RuntimeError(f"no repeated trunk class found (classes: "
                               f"{sorted((c, len(v)) for c, v in by_class.items())[:8]})")
    # Prefer the class living shallowest in the module tree: a block contains
    # its norm/attention submodules (which fire >= once per block and would win
    # by count), never the reverse. Tie-break by instance count.
    def _score(c):
        depth = np.mean([name.count(".") for name, _ in candidates[c]])
        return (depth, -len(candidates[c]))
    # Need enough instances to resolve d=0.1 - prefer fine-grained classes
    # (>= 8 units) over coarse stage-level containers; fall back if none.
    fine = {c: v for c, v in candidates.items() if len(v) >= 8}
    # Hybrid architectures (ViT-hybrid / DPT-hybrid): patch_embed wraps a CNN
    # backbone with its own repeated units. Choosing a single trunk class keeps
    # only the transformer blocks and resolves shallow depths to modules that
    # run AFTER the entire CNN. If repeated units exist both inside
    # patch_embed's backbone and outside it, chain them in execution order
    # instead: every backbone unit completes before any outer block, and each
    # candidate list is already in completion order.
    inside = {c: v for c, v in candidates.items()
              if all("patch_embed.backbone" in name for name, _ in v)}
    outside = {c: v for c, v in candidates.items()
               if all("patch_embed.backbone" not in name for name, _ in v)}
    if inside and outside:
        fine_in = {c: v for c, v in inside.items() if len(v) >= 8}
        back_cls = min(fine_in or inside, key=_score)
        fine_out = {c: v for c, v in outside.items() if len(v) >= 8}
        outer_cls = min(fine_out or outside, key=_score)
        chained = candidates[back_cls] + candidates[outer_cls]
        cls_by_name = {name: back_cls for name, _ in candidates[back_cls]}
        cls_by_name.update({name: outer_cls for name, _ in candidates[outer_cls]})
        n = len(chained)
        return [dict(module=name, cls=cls_by_name[name], n_blocks=n,
                     depth=(i + 1) / n, out_shape=str(shape))
                for i, (name, shape) in enumerate(chained)]
    trunk_cls = min(fine or candidates, key=_score)
    blocks = candidates[trunk_cls]
    n = len(blocks)
    return [dict(module=name, cls=trunk_cls, n_blocks=n, depth=(i + 1) / n,
                 out_shape=str(shape))
            for i, (name, shape) in enumerate(blocks)]


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    # the spec is the pipeline artifact the layer feature extraction reads back
    spec_csv = Path("experiments/rebuttal/universality_across_layers/spec/layer_spec_full.csv")
    spec_csv.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(Path(cfg.data_root) / "meta" / "things22k" / "model_metadata.csv")
    if cfg.models:
        meta = meta[meta["feature_uid"].isin(list(cfg.models))]

    prev = pd.read_csv(spec_csv) if spec_csv.exists() else pd.DataFrame()
    done_uids = set(prev["feature_uid"]) if len(prev) else set()
    rows, skips = ([] if prev.empty else prev.to_dict("records")), []

    for _, row in meta.iterrows():
        uid = row["feature_uid"]
        if uid in done_uids:
            continue
        try:
            model, transform = load_for_layers(row)
            model.eval()
            x = _dummy_input(transform)
            try:  # detectron2 builds on GPU when available - match the device
                x = x.to(next(model.parameters()).device)
            except StopIteration:
                pass
            blocks = trace_blocks(model, x)
            for t in TARGETS:
                best = min(blocks, key=lambda b: abs(b["depth"] - t))
                rows.append(dict(
                    feature_uid=uid, source=row["source"],
                    architecture=row.get("architecture", ""),
                    target_depth=t, achieved_depth=round(best["depth"], 4),
                    module=best["module"], trunk_class=best["cls"],
                    n_blocks=best["n_blocks"], out_shape=best["out_shape"],
                ))
            logger.info(f"{uid}: trunk={blocks[0]['cls']} n={blocks[0]['n_blocks']}")
            del model
            gc.collect()
        except Exception as e:
            logger.warning(f"SKIP {uid}: {e}")
            skips.append(dict(feature_uid=uid, source=row["source"],
                              error=f"{type(e).__name__}: {e}",
                              trace=traceback.format_exc()[-500:]))
        # checkpoint after every model so a crash loses nothing
        pd.DataFrame(rows).to_csv(spec_csv, index=False)
        if skips:
            pd.DataFrame(skips).to_csv(out / "skip_report.csv", index=False)

    logger.info(f"spec: {len(rows)} rows, skips: {len(skips)} -> {spec_csv}")


if __name__ == "__main__":
    main()
