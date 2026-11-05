from __future__ import annotations

from pathlib import Path


def resolve_things_image_path(
    raw_path: str | Path,
    images_root: Path,
    *,
    category: str | None = None,
    filename: str | None = None,
) -> Path:
    """Resolve a catalog row against a THINGS image root.

    The released catalog stores legacy paths such as
    ``images/things/<category>/<filename>``.
    """
    images_root = Path(images_root).expanduser()
    if category and filename:
        return images_root / str(category) / str(filename)

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if "things" in parts:
        parts = parts[parts.index("things") + 1 :]
    return images_root.joinpath(*parts)
