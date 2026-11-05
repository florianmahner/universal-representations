from pathlib import Path

from src.paths import resolve_things_image_path


def test_catalog_path_is_resolved_under_configured_image_root():
    root = Path("/datasets/things")

    result = resolve_things_image_path(
        "images/things/aardvark/aardvark_01b.jpg",
        images_root=root,
        category="aardvark",
        filename="aardvark_01b.jpg",
    )

    assert result == root / "aardvark" / "aardvark_01b.jpg"


def test_legacy_prefix_is_not_duplicated_without_catalog_columns():
    root = Path("/datasets/things")

    result = resolve_things_image_path(
        "images/things/aardvark/aardvark_01b.jpg",
        images_root=root,
    )

    assert result == root / "aardvark" / "aardvark_01b.jpg"
