from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA


def image_paths(data_root: Path) -> list[Path]:
    info = pd.read_csv(data_root / "meta" / "things22k" / "image_info_22k.csv")
    images = data_root / "images" / "things"
    return [images / cat / fn for cat, fn in zip(info["category"], info["filename"])]


def _load_image(path: Path, size: int, mode: str) -> np.ndarray:
    img = Image.open(path).convert(mode)
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def build_pixels(paths: list[Path], n_jobs: int, size: int = 224) -> np.ndarray:
    rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(lambda p: _load_image(p, size, "RGB").ravel())(p) for p in paths
    )
    return np.asarray(rows, dtype=np.float32)


def _gabor_bank(
    size: int = 128,
    scales: tuple[int, ...] = (4, 8, 16, 32),  # center frequencies in cycles per image
    n_orient: int = 8,
) -> np.ndarray:
    """Frequency-domain oriented Gaussian (Gabor) filters, (n_filters, size, size)."""
    fy = np.fft.fftfreq(size)[:, None] * size
    fx = np.fft.fftfreq(size)[None, :] * size
    radius = np.hypot(fx, fy)
    angle = np.arctan2(fy, fx)
    filters = []
    for f0 in scales:
        for k in range(n_orient):
            theta = np.pi * k / n_orient
            d_theta = np.angle(np.exp(1j * (angle - theta)))
            d_theta_m = np.angle(np.exp(1j * (angle - theta - np.pi)))
            radial = np.exp(-((radius - f0) ** 2) / (2 * (0.4 * f0) ** 2))
            angular = np.exp(-(d_theta**2) / (2 * (np.pi / n_orient) ** 2))
            angular_m = np.exp(-(d_theta_m**2) / (2 * (np.pi / n_orient) ** 2))
            filters.append(radial * (angular + angular_m))
    return np.asarray(filters, dtype=np.float32)


def _gabor_one(path: Path, bank: np.ndarray, pool: int = 4) -> np.ndarray:
    size = bank.shape[-1]
    img = _load_image(path, size, "L")
    spectrum = np.fft.fft2(img - img.mean())
    feats = []
    block = size // pool
    for filt in bank:
        energy = np.abs(np.fft.ifft2(spectrum * filt))
        pooled = energy.reshape(pool, block, pool, block).mean(axis=(1, 3))
        feats.append(pooled.ravel())
    return np.concatenate(feats).astype(np.float32)


def build_gabor(paths: list[Path], n_jobs: int) -> np.ndarray:
    bank = _gabor_bank()
    rows = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_gabor_one)(p, bank) for p in paths)
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    features = out / "features"
    features.mkdir(parents=True, exist_ok=True)
    paths = image_paths(Path(cfg.data_root))

    pixels_path = features / "pixels_rgb224.npy"
    if not pixels_path.exists():
        print(f"pixels_rgb224: {len(paths)} images")
        np.save(pixels_path, build_pixels(paths, cfg.n_jobs))
    pixels = np.load(pixels_path, mmap_mode="r")

    pca_path = features / "pixels_pca50.npy"
    if not pca_path.exists():
        print("pixels_pca50")
        np.save(pca_path, PCA(n_components=50, random_state=0).fit_transform(pixels).astype(np.float32))

    gabor_path = features / "gabor512.npy"
    if not gabor_path.exists():
        print("gabor512")
        np.save(gabor_path, build_gabor(paths, cfg.n_jobs))

    for f in sorted(features.glob("*.npy")):
        x = np.load(f, mmap_mode="r")
        print(f"{f.name}: {x.shape}")


if __name__ == "__main__":
    main()
