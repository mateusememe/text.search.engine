import json

from shotvault.models import Extraction, Kind, KeepValue, slugify_tag, wire_schema


def test_wire_schema_is_flat_and_strict():
    schema = wire_schema()
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["tags"]["maxItems"] == 8
    entity = schema["properties"]["entities"]["items"]
    assert entity["required"] == ["name", "type"]
    assert "person" in entity["properties"]["type"]["enum"]
    # the property called "description" survives annotation stripping
    assert "description" in schema["properties"]


def test_extraction_is_lenient_with_model_output():
    data = Extraction.model_validate(
        {
            "kind": "Tweet",
            "title": '"Screenshot of a cool trick."',
            "source_app": "unknown",
            "tags": "#Machine Learning, python, screenshot, 2024",
            "entities": ["Python", {"name": "python"}, {"name": "OpenAI", "type": "company"}],
            "actions": ["Read the paper", {"task": "", "due": "tomorrow"}, {"task": "Ship it", "due": "2025-01-31"}],
            "code": "```py\nprint('hi')\n```",
            "keep_value": "keep forever",
            "urls": ["https://github.com/x/y).", "not a url"],
        }
    )
    assert data.kind == Kind.SOCIAL_POST
    assert data.title == "a cool trick"
    assert data.source_app is None
    assert data.tags == ["machine-learning", "python"]
    assert [e.name for e in data.entities] == ["Python", "OpenAI"]
    assert data.entities[1].type.value == "other"
    assert [(a.task, a.due) for a in data.actions] == [("Read the paper", None), ("Ship it", "2025-01-31")]
    assert data.code == "print('hi')"
    assert data.keep_value == KeepValue.USEFUL
    assert data.urls == ["https://github.com/x/y"]


def test_slugify_tag_keeps_unicode_and_nesting():
    assert slugify_tag("#Programação Funcional") == "programação-funcional"
    assert slugify_tag("dev/Rust") == "dev/rust"
    assert slugify_tag("2024") == ""
