import importlib
import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]

OPTIONAL = {
    "torch", "torchvision", "timm", "tensorflow", "tensorflow_hub",
    "transformers", "open_clip", "clip", "detectron2", "visualpriors",
    "ftfy", "regex", "nltk", "sentence_transformers", "adjustText",
}


def _module_files():
    for base in ("src", "experiments"):
        for path in sorted((ROOT / base).rglob("*.py")):
            if "__pycache__" in path.parts or path.name.startswith("test_"):
                continue
            yield path


def _dotted_name(path: Path) -> str | None:
    parts = path.relative_to(ROOT).with_suffix("").parts
    for depth in range(1, len(parts)):
        if not (ROOT / Path(*parts[:depth]) / "__init__.py").exists():
            return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@pytest.mark.parametrize(
    "path", list(_module_files()), ids=lambda p: str(p.relative_to(ROOT))
)
def test_module_imports(path):
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    # Nearest ancestor first, so sibling-module imports (e.g. `from data
    # import ...` inside controlled_comparisons) win over the repo-root
    # data/ directory.
    chain = []
    for ancestor in [path.parent, *path.parent.parents]:
        chain.append(str(ancestor))
        if ancestor == ROOT:
            break
    sys.path[:0] = chain
    try:
        try:
            dotted = _dotted_name(path)
            if dotted:
                importlib.import_module(dotted)
            else:
                name = "smoke_" + "_".join(path.relative_to(ROOT).with_suffix("").parts)
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in OPTIONAL:
                pytest.skip(f"optional dependency missing: {exc.name}")
            raise
        except FileNotFoundError as exc:
            pytest.skip(f"requires data files: {exc}")
    finally:
        sys.path[:] = saved_path
        repo = str(ROOT)
        venv = str(ROOT / ".venv")
        for extra in set(sys.modules) - saved_modules:
            file = getattr(sys.modules[extra], "__file__", None)
            if file is None or (file.startswith(repo) and not file.startswith(venv)):
                del sys.modules[extra]
