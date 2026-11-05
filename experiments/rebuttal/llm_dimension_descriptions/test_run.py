from __future__ import annotations

import asyncio
import json

import pytest

from . import run
from .run import (
    CLASSIFICATION_SCHEMA,
    DESCRIPTION_SCHEMA,
    attach_universality,
    description_payload,
    description_bundle,
    derived_label,
    load_jsonl,
    parse_json_content,
    resolve_api_key,
    run_classifications,
    run_descriptions,
    summarize,
    validate_classification,
    validate_description,
)


def test_multiscale_payload_sends_text_then_five_images(tmp_path) -> None:
    image_paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        path.write_bytes(f"image-{index}".encode())
        image_paths.append(path)
    payload = description_payload(
        {"id": "vendor/model", "max_tokens": 100},
        image_paths,
        "overview_panels",
    )
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "do not count repetition" in content[0]["text"]
    assert [item["type"] for item in content[1:]] == ["image_url"] * 5


def test_transport_schemas_use_cross_provider_json_schema_subset() -> None:
    serialized = json.dumps([DESCRIPTION_SCHEMA, CLASSIFICATION_SCHEMA])
    for unsupported_keyword in [
        '"minimum"',
        '"maximum"',
        '"minItems"',
        '"maxItems"',
        '"uniqueItems"',
    ]:
        assert unsupported_keyword not in serialized


def test_description_validation_accepts_fenced_json() -> None:
    raw = {
        "patterns": [
            {
                "description": "elongated objects",
                "coverage": 0.75,
                "confidence": 0.8,
            }
        ],
        "overall_description": "Mostly long, narrow objects",
        "coherence": 0.7,
        "no_clear_pattern_probability": 0.1,
    }
    parsed = parse_json_content(f"```json\n{json.dumps(raw)}\n```")
    assert validate_description(parsed) == raw


def test_classification_probabilities_are_normalized() -> None:
    raw = {
        "semantic_strength": 0.7,
        "visual_strength": 0.6,
        "coherence": 0.8,
        "no_clear_pattern_probability": 0.1,
        "semantic_summary": "tools",
        "visual_summary": "elongated",
        "semantic_tags": ["object_category"],
        "semantic_domains": ["tools_instruments"],
        "visual_tags": ["shape_form"],
        "structural_tags": ["single_coherent_theme"],
        "mixed_relation": "independent",
        "source_agreement": 0.8,
        "four_way_probabilities": {
            "semantic": 0.1,
            "visual": 0.05,
            "mixed": 0.3,
            "neither": 0.05,
        },
        "rationale": "Both predicates are independently stated.",
    }
    result = validate_classification(raw)
    assert sum(result["four_way_probabilities"].values()) == pytest.approx(1)
    assert result["four_way_probabilities"]["mixed"] == pytest.approx(0.6)


def test_description_bundle_is_anonymized_and_deterministic() -> None:
    records = [
        {"requested_model": "vendor/a", "response": {"overall_description": "A"}},
        {"requested_model": "vendor/b", "response": {"overall_description": "B"}},
        {"requested_model": "vendor/c", "response": {"overall_description": "C"}},
    ]
    first, first_hash = description_bundle("model__d1", records)
    second, second_hash = description_bundle("model__d1", records)
    assert first == second
    assert first_hash == second_hash
    assert [item["source"] for item in first] == ["A", "B", "C"]
    assert "vendor/" not in json.dumps(first)


@pytest.mark.parametrize(
    ("semantic", "visual", "expected"),
    [
        (0.8, 0.2, "semantic"),
        (0.2, 0.8, "visual"),
        (0.8, 0.8, "mixed"),
        (0.2, 0.2, "neither"),
    ],
)
def test_derived_label(semantic: float, visual: float, expected: str) -> None:
    thresholds = {"semantic": 0.5, "visual": 0.5}
    assert derived_label(semantic, visual, thresholds) == expected


def test_universality_is_joined_from_current_metric_output(tmp_path) -> None:
    path = tmp_path / "universality.csv"
    path.write_text(
        "model,dimension,metric,universality\n"
        "example,0,cos2_perm,0.9\n"
        "example,0,cos2_perm_adj,0.25\n"
    )

    joined = attach_universality(
        [{"dim_id": "example__d0", "model": "example"}],
        path,
    )

    assert joined[0]["universality"] == "0.25"


def test_mocked_end_to_end_run_is_resumable(tmp_path, monkeypatch) -> None:
    async def direct_to_thread(function, /, *args, **kwargs):
        """Keep this unit test independent of the host's thread-pool support."""
        return function(*args, **kwargs)

    monkeypatch.setattr(run.asyncio, "to_thread", direct_to_thread)
    dimensions = [
        {
            "dim_id": "example__d0",
            "majority_label": "both",
            "universality": "0.3",
        }
    ]
    grids = tmp_path / "grids"
    grids.mkdir()
    (grids / "example__d0.png").write_bytes(b"fake png")
    output = tmp_path / "outputs"
    description_models = [
        {"id": "vendor/describe-a", "max_tokens": 50},
        {"id": "vendor/describe-b", "max_tokens": 50},
    ]
    classification_models = [{"id": "vendor/classify", "max_tokens": 50}]

    description_response = {
        "patterns": [
            {
                "description": "tools with elongated shapes",
                "coverage": 0.8,
                "confidence": 0.8,
            }
        ],
        "overall_description": "Mostly tools with elongated shapes",
        "coherence": 0.8,
        "no_clear_pattern_probability": 0.1,
    }
    classification_response = {
        "semantic_strength": 0.8,
        "visual_strength": 0.7,
        "coherence": 0.8,
        "no_clear_pattern_probability": 0.1,
        "semantic_summary": "tools",
        "visual_summary": "elongated shapes",
        "semantic_tags": ["object_category"],
        "semantic_domains": ["tools_instruments"],
        "visual_tags": ["shape_form"],
        "structural_tags": ["single_coherent_theme"],
        "mixed_relation": "independent",
        "source_agreement": 0.8,
        "four_way_probabilities": {
            "semantic": 0.1,
            "visual": 0.1,
            "mixed": 0.75,
            "neither": 0.05,
        },
        "rationale": "Independent category and shape properties.",
    }

    def fake_request(_url, payload=None, headers=None, timeout=180):
        del headers, timeout
        name = payload["response_format"]["json_schema"]["name"]
        response = (
            description_response
            if name == "dimension_description"
            else classification_response
        )
        return {
            "id": f"request-{name}",
            "model": payload["model"],
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.001,
            },
        }

    monkeypatch.setattr(run, "_request_json", fake_request)
    common = {
        "dimensions": dimensions,
        "output_dir": output,
        "api_key": "test",
        "catalog": {},
        "concurrency": 2,
        "max_retries": 1,
        "timeout": 1,
        "force": False,
    }
    asyncio.run(
        run_descriptions(
            **common,
            models=description_models,
            prompt_version="describe-v1",
            grid_dir=grids,
        )
    )
    assert len(load_jsonl(output / "descriptions.jsonl")) == 2

    # A resumed run must not append duplicate successful records.
    asyncio.run(
        run_descriptions(
            **common,
            models=description_models,
            prompt_version="describe-v1",
            grid_dir=grids,
        )
    )
    assert len(load_jsonl(output / "descriptions.jsonl")) == 2

    asyncio.run(
        run_classifications(
            **common,
            description_models=description_models,
            classification_models=classification_models,
            description_prompt_version="describe-v1",
            classification_prompt_version="classify-v1",
            minimum_descriptions=2,
        )
    )
    assert len(load_jsonl(output / "classifications.jsonl")) == 1

    config = {
        "description_models": description_models,
        "classification_models": classification_models,
        "prompt_versions": {
            "description": "describe-v1",
            "classification": "classify-v1",
        },
        "thresholds": {"semantic": 0.5, "visual": 0.5},
    }
    result = summarize(dimensions=dimensions, config=config, output_dir=output)
    assert result["dimensions_with_all_descriptions"] == 1
    assert result["dimensions_with_all_classifications"] == 1
    assert result["derived_label_counts"] == {"mixed": 1}
    assert result["exact_agreement_with_human_majority"] == 1
    assert result["actual_recorded_cost_usd"] == pytest.approx(0.003)


def test_api_key_path_accepts_dotenv_format(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENROUTER_API_KEY='test-key'\n")
    assert resolve_api_key(path) == "test-key"
