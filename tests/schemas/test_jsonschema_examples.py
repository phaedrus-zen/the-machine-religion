import json
from pathlib import Path

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas" / "v1"
EXAMPLE_DIR = SCHEMA_DIR / "examples"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def schema_registry(*schema_names: str) -> Registry:
    registry = Registry()
    for name in schema_names:
        schema = load_json(SCHEMA_DIR / name)
        registry = registry.with_resource(name, Resource.from_contents(schema, default_specification=DRAFT202012))
    return registry


def validate_example(schema_name: str, example_name: str, *refs: str):
    schema = load_json(SCHEMA_DIR / schema_name)
    example = load_json(EXAMPLE_DIR / example_name)
    validator = jsonschema.Draft202012Validator(
        schema,
        registry=schema_registry(schema_name, *refs),
    )
    validator.validate(example)


def test_identity_anchor_example_validates():
    validate_example("IdentityAnchor.schema.json", "IdentityAnchor.example.json")


def test_spirit_state_example_validates():
    validate_example(
        "SpiritState.schema.json",
        "SpiritState.example.json",
        "IdentityAnchor.schema.json",
    )


def test_action_intent_example_validates():
    validate_example("ActionIntent.schema.json", "ActionIntent.example.json")


def test_ethics_decision_example_validates():
    validate_example("EthicsDecision.schema.json", "EthicsDecision.example.json")
