"""Handler registry for ingestion connectors."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Type

from .base import IngestionHandler


class HandlerRegistry:
    """Central registry that maps handler names and filename patterns to classes."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Type[IngestionHandler]] = {}
        self._patterns: List[Tuple[str, str]] = []

    def register(
        self,
        name: str,
        handler: Type[IngestionHandler],
        patterns: Iterable[str] | None = None,
    ) -> None:
        if name in self._handlers:
            raise ValueError(f"Handler '{name}' is already registered")
        self._handlers[name] = handler
        for pattern in patterns or []:
            self._patterns.append((pattern, name))

    def get(self, name: str) -> Type[IngestionHandler]:
        try:
            return self._handlers[name]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise KeyError(f"Unknown handler '{name}'") from exc

    def match(self, path: Path) -> List[str]:
        resolved = str(path)
        return [name for pattern, name in self._patterns if fnmatch(resolved, pattern)]

    def create(self, name: str) -> IngestionHandler:
        handler_cls = self.get(name)
        return handler_cls()

    @property
    def names(self) -> List[str]:
        return sorted(self._handlers)


def get_registry() -> HandlerRegistry:
    """Singleton accessor used across the package."""

    if not hasattr(get_registry, "_instance"):
        get_registry._instance = HandlerRegistry()
    return get_registry._instance  # type: ignore[attr-defined]
