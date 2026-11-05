from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image


def load_image_info(data_root: Path = Path("data")) -> tuple[pd.DataFrame, list[str]]:
    info = pd.read_csv(data_root / "meta" / "things22k" / "image_info_22k.csv")
    concepts = sorted(info["category"].unique())
    return info, concepts


def concept_words(concepts: list[str], data_root: Path = Path("data")) -> list[str]:
    meta = pd.read_csv(
        data_root / "meta" / "things22k" / "concepts-metadata_things.tsv", sep="\t"
    )
    word_map = dict(zip(meta["uniqueID"], meta["Word"]))
    return [
        word_map.get(c, re.sub(r"\d+$", "", c).replace("_", " "))
        for c in concepts
    ]


def extract_bert(
    concepts: list[str],
    batch_size: int = 64,
    data_root: Path = Path("data"),
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    words = concept_words(concepts, data_root)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            enc = tok(
                words[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                return_special_tokens_mask=True,
            )
            special = enc.pop("special_tokens_mask")
            h = model(**enc).last_hidden_state
            mask = (enc["attention_mask"] * (1 - special)).unsqueeze(-1).float()
            embs.append((h * mask).sum(1) / mask.sum(1))
    return torch.cat(embs).numpy().astype(np.float32)


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def extract_untrained_conv1(
    info: pd.DataFrame,
    images_root: Path,
    batch_size: int = 256,
    num_workers: int = 32,
) -> np.ndarray:
    import torchvision.transforms as T
    from torchvision.models import resnet50

    torch.manual_seed(0)
    net = resnet50(weights=None)
    stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu).eval()

    tfm = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    paths = [
        images_root / cat / fname
        for cat, fname in zip(info["category"], info["filename"])
    ]
    loader = torch.utils.data.DataLoader(
        _ImageDataset(paths, tfm),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    feats = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            h = stem(x)
            feats.append(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1))
            if i % 10 == 0:
                print(f"conv1 batch {i + 1}/{len(loader)}", flush=True)
    return torch.cat(feats).numpy().astype(np.float32)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    data_root = Path(cfg.data_root)
    info, concepts = load_image_info(data_root)

    features = Path(__file__).parent / "outputs" / "features"
    features.mkdir(parents=True, exist_ok=True)
    (features / "concepts.json").write_text(json.dumps(concepts))

    bert_path = features / "bert_1854.npy"
    if not bert_path.exists():
        print(f"Extracting BERT embeddings for {len(concepts)} concepts", flush=True)
        np.save(bert_path, extract_bert(concepts, data_root=data_root))
    print(f"BERT: {np.load(bert_path).shape}", flush=True)

    conv1_path = features / "untrained_conv1_22k.npy"
    if not conv1_path.exists():
        print(f"Extracting untrained conv1 features for {len(info)} images", flush=True)
        np.save(
            conv1_path,
            extract_untrained_conv1(info, data_root / "images" / "things"),
        )
    print(f"conv1: {np.load(conv1_path).shape}", flush=True)


if __name__ == "__main__":
    main()
