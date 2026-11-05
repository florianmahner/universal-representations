from __future__ import annotations

from pathlib import Path

import numpy as np


def load_macaque(
    dataset: str,
    recording: str,
    roi: str = "it",
    min_reliab: float | None = 0.3,
    average_exemplars: bool = False,
    data_root: Path | str = Path("data"),
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load macaque neural data.

    Parameters
    ----------
    dataset : str
        "2k" (1854 categories, 1 exemplar) or "22k" (1854 categories, 12 exemplars).
    recording : str
        "F" (MonkeyF) or "N" (MonkeyN). For 2k, also "N1" or "N2".
    roi : str
        Brain region: "v1", "v4", or "it".
    min_reliab : float or None
        Filter channels by reliability threshold. None keeps all channels.
    average_exemplars : bool
        If True and dataset="22k", average across exemplars per category.
    data_root : Path or str
        Root of the data tree; the recordings live below
        ``data_root/features/macaque``.

    Returns
    -------
    data : ndarray of shape (n_stimuli, n_channels)
        Neural responses, filtered by reliability if specified.
    stimuli : list[str]
        Stimulus names (category names for 2k, exemplar names for 22k).
    reliab : ndarray of shape (n_channels,)
        Reliability values for returned channels.
    """
    path = (
        Path(data_root)
        / "features"
        / "macaque"
        / dataset
        / "processed"
        / f"{recording}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    d = np.load(path, allow_pickle=True).item()
    roi_slice = slice(*d["roi_slices"][roi])

    data = d["train_MUA"][roi_slice].T  # (n_stimuli, n_channels)
    reliab = d["reliab"]
    if reliab.ndim == 2:
        reliab = reliab.mean(axis=1)
    reliab = reliab[roi_slice]

    stimuli = list(d["stimuli"])

    if average_exemplars and dataset == "22k":
        categories = sorted(set(stimuli))
        data_avg = np.zeros((len(categories), data.shape[1]), dtype=data.dtype)
        for i, cat in enumerate(categories):
            mask = np.array([s == cat for s in stimuli])
            data_avg[i] = data[mask].mean(axis=0)
        data = data_avg
        stimuli = categories

    if min_reliab is not None:
        mask = reliab >= min_reliab
        data = data[:, mask]
        reliab = reliab[mask]

    return data, stimuli, reliab
