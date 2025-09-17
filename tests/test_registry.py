from pathlib import Path

from raging.ingest.registry import get_registry


def test_markdown_pattern_is_registered() -> None:
    registry = get_registry()
    matches = registry.match(Path("notes.md"))
    assert "markdown" in matches
