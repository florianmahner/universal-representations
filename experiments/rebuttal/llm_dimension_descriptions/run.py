#!/usr/bin/env python3
"""Describe and code NMF dimensions with an ensemble of vision language models.

Stage one shows each dimension's 8x8 image grid to several description models.
Stage two hands the anonymized, shuffled descriptions to a second ensemble that
codes them on the semantic and visual scales. Both stages append to JSONL logs
and resume from them, so an interrupted run never repeats a paid call.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Version stamped into every record written below; bump it when the meaning of
# a stored field changes, so old and new runs never get merged silently.
SCHEMA_VERSION = 1

PANEL_FILENAMES = [
    "panel_0_top_left.png",
    "panel_1_top_right.png",
    "panel_2_bottom_left.png",
    "panel_3_bottom_right.png",
]

SEMANTIC_TAGS = [
    "object_category",
    "superordinate_category",
    "function_affordance",
    "action_interaction",
    "scene_context",
    "part_whole",
    "relation_role",
    "material_identity",
    "other_semantic",
]
SEMANTIC_DOMAINS = [
    "animals",
    "plants",
    "food_drink",
    "body_people",
    "clothing",
    "tools_instruments",
    "vehicles",
    "furniture_household",
    "electronics",
    "natural_objects",
    "built_environment",
    "sports_leisure",
    "other",
]
VISUAL_TAGS = [
    "color_tone",
    "texture",
    "material_appearance",
    "shape_form",
    "spatial_layout",
    "viewpoint_orientation",
    "lighting",
    "scale_detail",
    "background_context",
    "photographic_style",
    "other_visual",
]
STRUCTURAL_TAGS = [
    "single_coherent_theme",
    "multiple_distinct_themes",
    "weak_partial_pattern",
    "polysemantic",
    "no_clear_pattern",
]
MIXED_RELATIONS = ["none", "independent", "category_induced", "ambiguous"]
LEGACY_LABELS = ["semantic", "visual", "mixed", "neither"]

DESCRIPTION_PROMPT = """\
You are documenting visible regularities in an image grid. The images were
selected because they receive high weights on one computational dimension.

Describe what the images share in ordinary, concrete language. Do not classify
the grid using broad scientific labels and do not speculate about the model,
the experiment, or why the images were grouped.

List zero to three independently noticeable shared properties. A property may
concern depicted items, their use or context, appearance, composition,
background, or any other regularity you can actually see. Strong object
similarity must not prevent you from noticing a separate appearance pattern.
Likewise, do not invent a pattern merely because some images happen to match.

For each property, estimate the fraction of the 64 images it covers and your
confidence. If there is no defensible shared property, return no patterns and
say so plainly. Keep every description concise and evidence-based."""

CLASSIFICATION_PROMPT = """\
You are coding anonymized, open-ended descriptions of one image-grid
dimension. You do not see the image grid, model identity, universality score,
or human ratings. Code only the evidence stated in the descriptions.

The grid was constructed from 64 high-weight dataset images, with at most one
image per dataset category, and displayed as equal-sized square thumbnails.
Judge dimension-specific, selective regularities: a high score requires a
property that meaningfully distinguishes this grid from a generic,
category-diverse sample of object-centric dataset images. Do not award
strength merely because the images contain recognizable content.

Semantic evidence concerns what is depicted or meant: identity, category,
function, action, scene, relation, or conceptual context. Visual evidence
concerns independently expressible appearance: color, texture, material
appearance, shape, layout, viewpoint, lighting, scale, background, or
photographic style.

For semantic strength, require a shared and reasonably specific semantic
predicate covering a substantial part of the grid. Merely being an
"everyday," "man-made," "physical," or "recognizable" object is too broad.
Lists of unrelated categories or functions indicate weak semantic coherence.
A superordinate concept such as clothing, food, musical instruments, or a
shared action can be strong even though the category-diversity rule prevents
exact category repetition.

For visual strength, require a concrete recurring appearance property.
Centered or isolated subjects, close crops, simple or neutral backgrounds,
one dominant object, equal-sized tiles, and generic product-, catalog-, or
fashion-style framing are common dataset or montage conventions. Treat them
as weak evidence unless the descriptions identify an unusually specific and
consistent variant. Specific form, color, texture, material appearance,
orientation, viewpoint, or background content may still be strong.

Mixed requires two strong, independently selective predicates. The mere
presence of both object words and photographic words in a description is not
mixed. Category-induced resemblance and generic presentation conventions must
not raise visual strength above the boundary. Likewise, a generic semantic
umbrella must not raise semantic strength above the boundary. For example,
heterogeneous objects sharing a circular form are visual, not mixed; people
wearing varied clothing in otherwise ordinary photographs are primarily
semantic; and tools consistently shown against green grass can be mixed.

Mark the mixed relation as:
- independent: both kinds of property are explicitly and independently stated;
- category_induced: visual resemblance merely follows from the category;
- ambiguous: both readings are plausible but not separable;
- none: there is no mixed relation.

Score semantic strength, visual strength, coherence, and no-pattern
probability from 0 to 1. Use 0.0-0.2 for absent or generic evidence, 0.3-0.49
for weak or non-selective evidence, 0.5-0.69 for a clear but partial property,
0.7-0.89 for strong evidence, and 0.9-1.0 only for a highly specific,
near-universal property. Code fine-grained tags, estimate agreement among the
descriptions, and provide probabilities for the four legacy outcomes. Base
those probabilities on the selective interpretation above rather than on the
mere co-occurrence of semantic and visual words. Probabilities must sum to 1.
The legacy key "mixed" corresponds to the old "both" category. In the
rationale, explicitly identify any generic cue that you discounted."""

DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "coverage": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": ["description", "coverage", "confidence"],
                "additionalProperties": False,
            },
        },
        "overall_description": {"type": "string"},
        "coherence": {"type": "number"},
        "no_clear_pattern_probability": {"type": "number"},
    },
    "required": [
        "patterns",
        "overall_description",
        "coherence",
        "no_clear_pattern_probability",
    ],
    "additionalProperties": False,
}

CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semantic_strength": {"type": "number"},
        "visual_strength": {"type": "number"},
        "coherence": {"type": "number"},
        "no_clear_pattern_probability": {"type": "number"},
        "semantic_summary": {"type": "string"},
        "visual_summary": {"type": "string"},
        "semantic_tags": {
            "type": "array",
            "items": {"type": "string", "enum": SEMANTIC_TAGS},
        },
        "semantic_domains": {
            "type": "array",
            "items": {"type": "string", "enum": SEMANTIC_DOMAINS},
        },
        "visual_tags": {
            "type": "array",
            "items": {"type": "string", "enum": VISUAL_TAGS},
        },
        "structural_tags": {
            "type": "array",
            "items": {"type": "string", "enum": STRUCTURAL_TAGS},
        },
        "mixed_relation": {"type": "string", "enum": MIXED_RELATIONS},
        "source_agreement": {"type": "number"},
        "four_way_probabilities": {
            "type": "object",
            "properties": {label: {"type": "number"} for label in LEGACY_LABELS},
            "required": LEGACY_LABELS,
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "semantic_strength",
        "visual_strength",
        "coherence",
        "no_clear_pattern_probability",
        "semantic_summary",
        "visual_summary",
        "semantic_tags",
        "semantic_domains",
        "visual_tags",
        "structural_tags",
        "mixed_relation",
        "source_agreement",
        "four_way_probabilities",
        "rationale",
    ],
    "additionalProperties": False,
}


class OpenRouterError(RuntimeError):
    """An OpenRouter request failed."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp01(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return min(1.0, max(0.0, number))


def parse_json_content(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def validate_description(value: dict[str, Any]) -> dict[str, Any]:
    patterns = value.get("patterns")
    if not isinstance(patterns, list) or len(patterns) > 3:
        raise ValueError("patterns must be a list with at most three entries")
    clean_patterns = []
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            raise ValueError(f"patterns[{index}] must be an object")
        description = str(pattern.get("description", "")).strip()
        if not description:
            raise ValueError(f"patterns[{index}].description is empty")
        clean_patterns.append(
            {
                "description": description,
                "coverage": clamp01(pattern.get("coverage"), "coverage"),
                "confidence": clamp01(pattern.get("confidence"), "confidence"),
            }
        )
    overall = str(value.get("overall_description", "")).strip()
    if not overall:
        raise ValueError("overall_description is empty")
    return {
        "patterns": clean_patterns,
        "overall_description": overall,
        "coherence": clamp01(value.get("coherence"), "coherence"),
        "no_clear_pattern_probability": clamp01(
            value.get("no_clear_pattern_probability"),
            "no_clear_pattern_probability",
        ),
    }


def _clean_enum_list(value: Any, allowed: list[str], field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    invalid = sorted(set(map(str, value)).difference(allowed))
    if invalid:
        raise ValueError(f"{field} contains invalid values: {invalid}")
    return list(dict.fromkeys(map(str, value)))


def validate_classification(value: dict[str, Any]) -> dict[str, Any]:
    probabilities = value.get("four_way_probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("four_way_probabilities must be an object")
    clean_probabilities = {
        label: clamp01(probabilities.get(label), f"probability.{label}")
        for label in LEGACY_LABELS
    }
    total = sum(clean_probabilities.values())
    if total <= 0:
        raise ValueError("four-way probabilities sum to zero")
    clean_probabilities = {
        label: probability / total for label, probability in clean_probabilities.items()
    }
    relation = str(value.get("mixed_relation", ""))
    if relation not in MIXED_RELATIONS:
        raise ValueError(f"invalid mixed_relation: {relation!r}")
    result = {
        field: clamp01(value.get(field), field)
        for field in [
            "semantic_strength",
            "visual_strength",
            "coherence",
            "no_clear_pattern_probability",
            "source_agreement",
        ]
    }
    result.update(
        {
            "semantic_summary": str(value.get("semantic_summary", "")).strip(),
            "visual_summary": str(value.get("visual_summary", "")).strip(),
            "semantic_tags": _clean_enum_list(
                value.get("semantic_tags"), SEMANTIC_TAGS, "semantic_tags"
            ),
            "semantic_domains": _clean_enum_list(
                value.get("semantic_domains"),
                SEMANTIC_DOMAINS,
                "semantic_domains",
            ),
            "visual_tags": _clean_enum_list(
                value.get("visual_tags"), VISUAL_TAGS, "visual_tags"
            ),
            "structural_tags": _clean_enum_list(
                value.get("structural_tags"),
                STRUCTURAL_TAGS,
                "structural_tags",
            ),
            "mixed_relation": relation,
            "four_way_probabilities": clean_probabilities,
            "rationale": str(value.get("rationale", "")).strip(),
        }
    )
    return result


def structured_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {path}")
    for role in ["description_models", "classification_models"]:
        models = config.get(role)
        if not isinstance(models, list) or not models:
            raise ValueError(f"{role} must be a non-empty list")
        ids = [model.get("id") for model in models]
        if any(not model_id for model_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{role} contains missing or duplicate model IDs")
    return config


def load_dimensions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len({row["dim_id"] for row in rows}):
        raise ValueError(f"Duplicate dim_id values in {path}")
    return rows


def attach_universality(
    dimensions: list[dict[str, str]],
    path: Path,
    *,
    metric: str = "cos2_perm_adj",
) -> list[dict[str, str]]:
    """Join freshly computed rank-50 universality to the rating aggregate."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Universality scores not found: {path}. Run "
            "`python -m experiments.universality.metric.compute.run` "
            "before the interpretation analysis."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    scores: dict[str, str] = {}
    for row in rows:
        if row.get("metric") != metric:
            continue
        try:
            dim_id = f"{row['model']}__d{int(row['dimension'])}"
            value = float(row["universality"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed universality row in {path}: {row}") from error
        if not math.isfinite(value):
            raise ValueError(f"Non-finite universality for {dim_id} in {path}")
        if dim_id in scores:
            raise ValueError(
                f"Duplicate {metric} universality score for {dim_id} in {path}"
            )
        scores[dim_id] = f"{value:.17g}"

    joined: list[dict[str, str]] = []
    missing: list[str] = []
    for row in dimensions:
        dim_id = row["dim_id"]
        if dim_id not in scores:
            missing.append(dim_id)
            continue
        joined.append({**row, "universality": scores[dim_id]})
    if missing:
        raise ValueError(
            f"{len(missing)} rated dimensions lack {metric} universality scores "
            f"in {path}; first: {missing[:5]}"
        )
    return joined


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def description_bundle(
    dim_id: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    responses = [record["response"] for record in records]
    seed = int(hashlib.sha256(dim_id.encode()).hexdigest()[:16], 16)
    order = list(range(len(responses)))
    random.Random(seed).shuffle(order)
    bundle = [
        {"source": chr(ord("A") + index), "description": responses[source_index]}
        for index, source_index in enumerate(order)
    ]
    return bundle, sha256_json(bundle)


def latest_records(
    records: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]
) -> dict[Any, dict[str, Any]]:
    return {key(record): record for record in records}


def _request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 180,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise OpenRouterError(
            f"OpenRouter HTTP {error.code}: {body[:1000]}", status=error.code
        ) from error
    except urllib.error.URLError as error:
        raise OpenRouterError(f"OpenRouter network error: {error}") from error


def openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "anonymous submission analysis",
    }


def resolve_api_key(explicit_path: Path | None = None) -> str:
    if value := os.environ.get("OPENROUTER_API_KEY"):
        return value.strip()
    candidates = [
        explicit_path,
        Path(__file__).parent / ".env",
        Path.home() / ".config" / "openrouter" / "api_key.txt",
    ]
    for path in candidates:
        if path and path.exists():
            text = path.read_text().strip()
            for line in text.splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip("\"'")
            return text
    raise FileNotFoundError(
        "Set OPENROUTER_API_KEY, or write it to a .env file beside this script. "
        "No paid call was made."
    )


def fetch_catalog() -> dict[str, dict[str, Any]]:
    response = _request_json("https://openrouter.ai/api/v1/models", timeout=60)
    return {model["id"]: model for model in response.get("data", [])}


def pricing_for(
    requested_model: str,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, float]:
    pricing = catalog.get(requested_model, {}).get("pricing", {})
    result = {}
    for key in ["prompt", "completion", "image", "internal_reasoning"]:
        try:
            result[key] = float(pricing.get(key, 0))
        except (TypeError, ValueError):
            result[key] = 0.0
    return result


def usage_cost(
    usage: dict[str, Any],
    requested_model: str,
    resolved_model: str,
    catalog: dict[str, dict[str, Any]],
) -> float | None:
    if usage.get("cost") is not None:
        try:
            return float(usage["cost"])
        except (TypeError, ValueError):
            pass
    pricing = pricing_for(resolved_model, catalog)
    if not any(pricing.values()):
        pricing = pricing_for(requested_model, catalog)
    if not any(pricing.values()):
        return None
    prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = float(usage.get("completion_tokens", 0) or 0)
    return prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]


def model_payload_options(model: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"max_tokens": int(model["max_tokens"])}
    if "temperature" in model:
        options["temperature"] = float(model["temperature"])
    if effort := model.get("reasoning_effort"):
        options["reasoning"] = {"effort": effort, "exclude": True}
    return options


def description_payload(
    model: dict[str, Any],
    image_paths: list[Path],
    image_mode: str,
) -> dict[str, Any]:
    if image_mode == "grid":
        instruction = "Describe the shared properties in this 8×8 grid."
    elif image_mode == "overview_panels":
        instruction = (
            "Describe the shared properties across these 64 images. The first "
            "image is an 8×8 overview. The next four images are higher-resolution "
            "4×4 views of its top-left, top-right, bottom-left, and bottom-right "
            "quadrants. They repeat the same images, so do not count repetition "
            "as additional evidence."
        )
    else:
        raise ValueError(f"Unsupported description image mode: {image_mode}")
    image_content = []
    for image_path in image_paths:
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        image_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    return {
        "model": model["id"],
        **model_payload_options(model),
        "provider": {"require_parameters": True},
        "response_format": structured_format(
            "dimension_description", DESCRIPTION_SCHEMA
        ),
        "messages": [
            {"role": "system", "content": DESCRIPTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    },
                    *image_content,
                ],
            },
        ],
    }


def classification_payload(
    model: dict[str, Any],
    bundle: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model["id"],
        **model_payload_options(model),
        "provider": {"require_parameters": True},
        "response_format": structured_format(
            "dimension_classification", CLASSIFICATION_SCHEMA
        ),
        "messages": [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {
                "role": "user",
                "content": (
                    "Code these independent descriptions:\n"
                    + json.dumps(bundle, ensure_ascii=False)
                ),
            },
        ],
    }


def response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Missing message content: {response}") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        if parts:
            return "".join(parts)
    raise ValueError(f"Unsupported message content: {content!r}")


async def request_with_retries(
    *,
    payload_factory: Callable[[], dict[str, Any]],
    api_key: str,
    semaphore: asyncio.Semaphore,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    max_retries: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    retryable = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(max_retries):
        started = time.monotonic()
        try:
            async with semaphore:
                payload = await asyncio.to_thread(payload_factory)
                response = await asyncio.to_thread(
                    _request_json,
                    "https://openrouter.ai/api/v1/chat/completions",
                    payload,
                    openrouter_headers(api_key),
                    timeout,
                )
            parsed = parse_json_content(response_text(response))
            return validator(parsed), response, time.monotonic() - started
        except OpenRouterError as error:
            if error.status not in retryable or attempt + 1 == max_retries:
                raise
        except (json.JSONDecodeError, ValueError):
            if attempt + 1 == max_retries:
                raise
        await asyncio.sleep(min(30, 2**attempt))
    raise AssertionError("retry loop exhausted")


def completed_description_keys(
    records: list[dict[str, Any]], prompt_version: str, image_mode: str
) -> set[tuple[str, str]]:
    return {
        (record["dim_id"], record["requested_model"])
        for record in records
        if record.get("prompt_version") == prompt_version
        and record.get("image_mode", "grid") == image_mode
    }


def completed_classification_keys(
    records: list[dict[str, Any]], prompt_version: str
) -> set[tuple[str, str, str]]:
    return {
        (record["dim_id"], record["requested_model"], record["input_hash"])
        for record in records
        if record.get("prompt_version") == prompt_version
    }


def description_image_paths(
    *,
    dim_id: str,
    image_mode: str,
    grid_dir: Path,
    panel_dir: Path,
) -> list[Path]:
    overview = grid_dir / f"{dim_id}.png"
    if image_mode == "grid":
        return [overview]
    if image_mode == "overview_panels":
        return [overview] + [
            panel_dir / dim_id / filename for filename in PANEL_FILENAMES
        ]
    raise ValueError(f"Unsupported description image mode: {image_mode}")


async def run_descriptions(
    *,
    dimensions: list[dict[str, str]],
    models: list[dict[str, Any]],
    prompt_version: str,
    grid_dir: Path,
    output_dir: Path,
    api_key: str,
    catalog: dict[str, dict[str, Any]],
    concurrency: int = 8,
    max_retries: int = 4,
    timeout: float = 180,
    force: bool = False,
    image_mode: str = "grid",
    panel_dir: Path = Path(__file__).parent / "outputs" / "panel_cache_dimensions",
) -> None:
    output = output_dir / "descriptions.jsonl"
    errors = output_dir / "errors.jsonl"
    done = completed_description_keys(load_jsonl(output), prompt_version, image_mode)
    jobs = [
        (row, model)
        for row in dimensions
        for model in models
        if force or (row["dim_id"], model["id"]) not in done
    ]
    logger.info(
        "description stage: %d pending / %d configured calls",
        len(jobs),
        len(dimensions) * len(models),
    )
    if not jobs:
        return
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counts = Counter()

    async def work(row: dict[str, str], model: dict[str, Any]) -> None:
        dim_id = row["dim_id"]
        try:
            image_paths = description_image_paths(
                dim_id=dim_id,
                image_mode=image_mode,
                grid_dir=grid_dir,
                panel_dir=panel_dir,
            )
            validated, raw, latency = await request_with_retries(
                payload_factory=lambda: description_payload(
                    model, image_paths, image_mode
                ),
                api_key=api_key,
                semaphore=semaphore,
                validator=validate_description,
                max_retries=max_retries,
                timeout=timeout,
            )
            usage = raw.get("usage", {})
            resolved_model = raw.get("model", model["id"])
            record = {
                "schema_version": SCHEMA_VERSION,
                "stage": "description",
                "prompt_version": prompt_version,
                "image_mode": image_mode,
                "n_input_images": len(image_paths),
                "dim_id": dim_id,
                "requested_model": model["id"],
                "resolved_model": resolved_model,
                "created_at_utc": utc_now(),
                "response": validated,
                "usage": usage,
                "cost_usd": usage_cost(usage, model["id"], resolved_model, catalog),
                "latency_seconds": latency,
                "request_id": raw.get("id"),
            }
            async with write_lock:
                await asyncio.to_thread(append_jsonl, output, record)
            counts["ok"] += 1
        except Exception as error:
            record = {
                "schema_version": SCHEMA_VERSION,
                "stage": "description",
                "prompt_version": prompt_version,
                "image_mode": image_mode,
                "dim_id": dim_id,
                "requested_model": model["id"],
                "created_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            async with write_lock:
                await asyncio.to_thread(append_jsonl, errors, record)
            counts["error"] += 1
        completed = counts["ok"] + counts["error"]
        if completed % 25 == 0 or completed == len(jobs):
            logger.info(
                "description progress: %d/%d (%d ok, %d errors)",
                completed,
                len(jobs),
                counts["ok"],
                counts["error"],
            )

    await asyncio.gather(*(work(row, model) for row, model in jobs))


def current_descriptions(
    records: list[dict[str, Any]],
    dimensions: list[dict[str, str]],
    model_ids: list[str],
    prompt_version: str,
    image_mode: str = "grid",
) -> dict[str, list[dict[str, Any]]]:
    selected = latest_records(
        [
            record
            for record in records
            if record.get("prompt_version") == prompt_version
            and record.get("requested_model") in model_ids
            and record.get("image_mode", "grid") == image_mode
        ],
        key=lambda record: (record["dim_id"], record["requested_model"]),
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for row in dimensions:
        dim_id = row["dim_id"]
        result[dim_id] = [
            selected[(dim_id, model_id)]
            for model_id in model_ids
            if (dim_id, model_id) in selected
        ]
    return result


async def run_classifications(
    *,
    dimensions: list[dict[str, str]],
    description_models: list[dict[str, Any]],
    classification_models: list[dict[str, Any]],
    description_prompt_version: str,
    classification_prompt_version: str,
    minimum_descriptions: int,
    output_dir: Path,
    api_key: str,
    catalog: dict[str, dict[str, Any]],
    concurrency: int = 8,
    max_retries: int = 4,
    timeout: float = 180,
    force: bool = False,
    description_image_mode: str = "grid",
) -> None:
    description_records = load_jsonl(output_dir / "descriptions.jsonl")
    descriptions = current_descriptions(
        description_records,
        dimensions,
        [model["id"] for model in description_models],
        description_prompt_version,
        description_image_mode,
    )
    output = output_dir / "classifications.jsonl"
    errors = output_dir / "errors.jsonl"
    done = completed_classification_keys(
        load_jsonl(output), classification_prompt_version
    )
    jobs = []
    missing = 0
    for row in dimensions:
        dim_id = row["dim_id"]
        if len(descriptions[dim_id]) < minimum_descriptions:
            missing += 1
            continue
        bundle, input_hash = description_bundle(dim_id, descriptions[dim_id])
        for model in classification_models:
            key = (dim_id, model["id"], input_hash)
            if force or key not in done:
                jobs.append((row, model, bundle, input_hash))
    logger.info(
        "classification stage: %d pending calls; %d dimensions lack %d descriptions",
        len(jobs),
        missing,
        minimum_descriptions,
    )
    if not jobs:
        return
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counts = Counter()

    async def work(
        row: dict[str, str],
        model: dict[str, Any],
        bundle: list[dict[str, Any]],
        input_hash: str,
    ) -> None:
        dim_id = row["dim_id"]
        try:
            validated, raw, latency = await request_with_retries(
                payload_factory=lambda: classification_payload(model, bundle),
                api_key=api_key,
                semaphore=semaphore,
                validator=validate_classification,
                max_retries=max_retries,
                timeout=timeout,
            )
            usage = raw.get("usage", {})
            resolved_model = raw.get("model", model["id"])
            record = {
                "schema_version": SCHEMA_VERSION,
                "stage": "classification",
                "prompt_version": classification_prompt_version,
                "description_prompt_version": description_prompt_version,
                "description_image_mode": description_image_mode,
                "dim_id": dim_id,
                "requested_model": model["id"],
                "resolved_model": resolved_model,
                "input_hash": input_hash,
                "n_descriptions": len(bundle),
                "created_at_utc": utc_now(),
                "response": validated,
                "usage": usage,
                "cost_usd": usage_cost(usage, model["id"], resolved_model, catalog),
                "latency_seconds": latency,
                "request_id": raw.get("id"),
            }
            async with write_lock:
                await asyncio.to_thread(append_jsonl, output, record)
            counts["ok"] += 1
        except Exception as error:
            record = {
                "schema_version": SCHEMA_VERSION,
                "stage": "classification",
                "prompt_version": classification_prompt_version,
                "description_prompt_version": description_prompt_version,
                "description_image_mode": description_image_mode,
                "dim_id": dim_id,
                "requested_model": model["id"],
                "input_hash": input_hash,
                "created_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            async with write_lock:
                await asyncio.to_thread(append_jsonl, errors, record)
            counts["error"] += 1
        completed = counts["ok"] + counts["error"]
        if completed % 25 == 0 or completed == len(jobs):
            logger.info(
                "classification progress: %d/%d (%d ok, %d errors)",
                completed,
                len(jobs),
                counts["ok"],
                counts["error"],
            )

    await asyncio.gather(
        *(
            work(row, model, bundle, input_hash)
            for row, model, bundle, input_hash in jobs
        )
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def derived_label(semantic: float, visual: float, thresholds: dict[str, Any]) -> str:
    has_semantic = semantic >= float(thresholds["semantic"])
    has_visual = visual >= float(thresholds["visual"])
    if has_semantic and has_visual:
        return "mixed"
    if has_semantic:
        return "semantic"
    if has_visual:
        return "visual"
    return "neither"


def summarize(
    *,
    dimensions: list[dict[str, str]],
    config: dict[str, Any],
    output_dir: Path,
    description_image_mode: str = "grid",
    universality_source: Path | None = None,
) -> dict[str, Any]:
    description_version = config["prompt_versions"]["description"]
    classification_version = config["prompt_versions"]["classification"]
    description_model_ids = [model["id"] for model in config["description_models"]]
    classification_model_ids = [
        model["id"] for model in config["classification_models"]
    ]
    descriptions = current_descriptions(
        load_jsonl(output_dir / "descriptions.jsonl"),
        dimensions,
        description_model_ids,
        description_version,
        description_image_mode,
    )
    raw_classifications = load_jsonl(output_dir / "classifications.jsonl")
    latest_classifications = latest_records(
        [
            record
            for record in raw_classifications
            if record.get("prompt_version") == classification_version
            and record.get("requested_model") in classification_model_ids
        ],
        key=lambda record: (
            record["dim_id"],
            record["requested_model"],
            record["input_hash"],
        ),
    )

    summary_rows = []
    tag_rows = []
    label_counts = Counter()
    human_matches = []
    for row in dimensions:
        dim_id = row["dim_id"]
        desc = descriptions[dim_id]
        bundle, input_hash = description_bundle(dim_id, desc) if desc else ([], "")
        classifications = [
            latest_classifications[(dim_id, model_id, input_hash)]
            for model_id in classification_model_ids
            if (dim_id, model_id, input_hash) in latest_classifications
        ]
        responses = [record["response"] for record in classifications]
        semantic = mean([response["semantic_strength"] for response in responses])
        visual = mean([response["visual_strength"] for response in responses])
        output_row: dict[str, Any] = {
            **row,
            "n_descriptions": len(desc),
            "n_classifications": len(responses),
            "description_input_hash": input_hash,
            "descriptions": json.dumps(
                [record["response"]["overall_description"] for record in desc],
                ensure_ascii=False,
            ),
        }
        if responses:
            label = derived_label(semantic, visual, config["thresholds"])
            output_row.update(
                {
                    "semantic_strength": semantic,
                    "visual_strength": visual,
                    "coherence": mean(
                        [response["coherence"] for response in responses]
                    ),
                    "no_clear_pattern_probability": mean(
                        [
                            response["no_clear_pattern_probability"]
                            for response in responses
                        ]
                    ),
                    "source_agreement": mean(
                        [response["source_agreement"] for response in responses]
                    ),
                    "semantic_strength_sd": sample_std(
                        [response["semantic_strength"] for response in responses]
                    ),
                    "visual_strength_sd": sample_std(
                        [response["visual_strength"] for response in responses]
                    ),
                    "derived_label": label,
                }
            )
            for legacy_label in LEGACY_LABELS:
                output_row[f"probability_{legacy_label}"] = mean(
                    [
                        response["four_way_probabilities"][legacy_label]
                        for response in responses
                    ]
                )
            for group, allowed in [
                ("semantic_tag", SEMANTIC_TAGS),
                ("semantic_domain", SEMANTIC_DOMAINS),
                ("visual_tag", VISUAL_TAGS),
                ("structural_tag", STRUCTURAL_TAGS),
            ]:
                response_field = {
                    "semantic_tag": "semantic_tags",
                    "semantic_domain": "semantic_domains",
                    "visual_tag": "visual_tags",
                    "structural_tag": "structural_tags",
                }[group]
                for tag in allowed:
                    prevalence = mean(
                        [
                            float(tag in response[response_field])
                            for response in responses
                        ]
                    )
                    output_row[f"{group}__{tag}"] = prevalence
                    if prevalence > 0:
                        tag_rows.append(
                            {
                                "dim_id": dim_id,
                                "tag_group": group,
                                "tag": tag,
                                "coder_prevalence": prevalence,
                                "universality": row.get("universality", ""),
                            }
                        )
            label_counts[label] += 1
            human = (
                "mixed"
                if row.get("majority_label") == "both"
                else row.get("majority_label")
            )
            if human:
                human_matches.append(float(human == label))
        summary_rows.append(output_row)

    all_records = load_jsonl(output_dir / "descriptions.jsonl") + load_jsonl(
        output_dir / "classifications.jsonl"
    )
    usage_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        usage_groups[(record["stage"], record["requested_model"])].append(record)
    usage_rows = []
    for (stage, model), records in sorted(usage_groups.items()):
        costs = [
            float(record["cost_usd"])
            for record in records
            if record.get("cost_usd") is not None
        ]
        usage_rows.append(
            {
                "stage": stage,
                "model": model,
                "requests": len(records),
                "prompt_tokens": sum(
                    int(record.get("usage", {}).get("prompt_tokens", 0) or 0)
                    for record in records
                ),
                "completion_tokens": sum(
                    int(record.get("usage", {}).get("completion_tokens", 0) or 0)
                    for record in records
                ),
                "cost_usd": sum(costs) if costs else "",
            }
        )

    atomic_write_csv(output_dir / "dimension_summary.csv", summary_rows)
    atomic_write_csv(output_dir / "dimension_tags.csv", tag_rows)
    atomic_write_csv(output_dir / "usage_summary.csv", usage_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "n_dimensions": len(dimensions),
        "dimensions_with_all_descriptions": sum(
            len(descriptions[row["dim_id"]]) == len(description_model_ids)
            for row in dimensions
        ),
        "dimensions_with_all_classifications": sum(
            int(summary_row["n_classifications"]) == len(classification_model_ids)
            for summary_row in summary_rows
        ),
        "derived_label_counts": dict(label_counts),
        "exact_agreement_with_human_majority": (
            mean(human_matches) if human_matches else None
        ),
        "actual_recorded_cost_usd": sum(
            float(row["cost_usd"]) for row in usage_rows if row["cost_usd"] != ""
        ),
        "thresholds": config["thresholds"],
        "prompt_versions": config["prompt_versions"],
        "description_image_mode": description_image_mode,
        "universality_source": (
            str(universality_source.resolve())
            if universality_source is not None
            else None
        ),
    }
    atomic_write_json(output_dir / "summary.json", result)
    logger.info("summary written to %s", output_dir)
    return result


def validate_inputs(
    dimensions: list[dict[str, str]],
    grid_dir: Path,
    image_mode: str = "grid",
    panel_dir: Path = Path(__file__).parent / "outputs" / "panel_cache_dimensions",
) -> None:
    missing = []
    for row in dimensions:
        paths = description_image_paths(
            dim_id=row["dim_id"],
            image_mode=image_mode,
            grid_dir=grid_dir,
            panel_dir=panel_dir,
        )
        if not all(path.is_file() for path in paths):
            missing.append(row["dim_id"])
    if missing:
        raise FileNotFoundError(
            f"Missing image inputs for {len(missing)} dimensions in "
            f"{image_mode} mode; first: {missing[:5]}"
        )


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    config = load_config(Path(__file__).parent / "models_published.json")
    grid_dir = Path("experiments/universality/notebooks/grid_cache_dimensions")
    universality = (
        Path("experiments/universality/shared/data")
        / f"universality_{cfg.dataset}_rank_{cfg.factorization.rank}.csv"
    )
    dimensions = attach_universality(
        load_dimensions(
            Path(cfg.data_root) / "ratings" / "dim_ratings_aggregated_qc_attn_catch.csv"
        ),
        universality,
    )
    validate_inputs(dimensions, grid_dir)

    catalog = fetch_catalog()
    atomic_write_json(
        out / "openrouter_model_catalog.json",
        {"fetched_at_utc": utc_now(), "data": list(catalog.values())},
    )

    api_key = resolve_api_key()
    asyncio.run(
        run_descriptions(
            dimensions=dimensions,
            models=config["description_models"],
            prompt_version=config["prompt_versions"]["description"],
            grid_dir=grid_dir,
            output_dir=out,
            api_key=api_key,
            catalog=catalog,
        )
    )
    asyncio.run(
        run_classifications(
            dimensions=dimensions,
            description_models=config["description_models"],
            classification_models=config["classification_models"],
            description_prompt_version=config["prompt_versions"]["description"],
            classification_prompt_version=config["prompt_versions"]["classification"],
            minimum_descriptions=int(config["minimum_descriptions"]),
            output_dir=out,
            api_key=api_key,
            catalog=catalog,
        )
    )
    summarize(
        dimensions=dimensions,
        config=config,
        output_dir=out,
        universality_source=universality,
    )


if __name__ == "__main__":
    main()
