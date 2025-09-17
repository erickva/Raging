"""High-level ingestion pipeline orchestration."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Mapping, Optional

from ..config import ProjectConfig, SourceConfig
from .base import DocumentChunk, SourceDocument, to_source_id
from .handlers import register_default_handlers
from .registry import HandlerRegistry, get_registry


@dataclass
class IngestionStats:
    processed: int = 0
    skipped: int = 0
    emitted: int = 0


class IngestionPipeline:
    """Drive the end-to-end ingestion flow for configured sources."""

    def __init__(
        self,
        config: ProjectConfig,
        registry: HandlerRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or get_registry()
        register_default_handlers()
        self.stats = IngestionStats()
        self._document_stats: dict[str, IngestionStats] = {}
        self._documents: dict[str, SourceDocument] = {}

    def plan_documents(self) -> List[SourceDocument]:
        """Return a list of documents that will be processed."""

        documents: List[SourceDocument] = []
        for source_cfg in self.config.ingest.sources:
            explicit_files = source_cfg.files
            if explicit_files:
                for path in explicit_files:
                    documents.append(
                        SourceDocument(
                            source_id=to_source_id(path),
                            path=path,
                            metadata={"collection": self.config.storage.collection, **source_cfg.metadata},
                            tags=tuple(source_cfg.tags),
                        )
                    )
                continue
            if not source_cfg.path:
                continue
            for path in self._enumerate_paths(source_cfg):
                documents.append(
                    SourceDocument(
                        source_id=to_source_id(path),
                        path=path,
                        metadata={"collection": self.config.storage.collection, **source_cfg.metadata},
                        tags=tuple(source_cfg.tags),
                    )
                )
        return documents

    def run(
        self,
        *,
        existing_checksums: Optional[Mapping[str, str]] = None,
        force: bool = False,
    ) -> Iterator[DocumentChunk]:
        """Yield chunks for all configured sources."""

        known = dict(existing_checksums or {})
        checksum_algo = self.config.ingest.checksum_algorithm
        self.stats = IngestionStats()
        self._document_stats = {}
        self._documents = {}
        for document in self.plan_documents():
            handler = self._resolve_handler(document.path)
            doc_stats = IngestionStats()
            self._document_stats[document.source_id] = doc_stats
            self._documents[document.source_id] = document
            for chunk in handler.iter_chunks(document, checksum_algo):
                self.stats.processed += 1
                doc_stats.processed += 1
                if not force:
                    existing_checksum = known.get(chunk.chunk_id)
                    if existing_checksum == chunk.checksum:
                        self.stats.skipped += 1
                        doc_stats.skipped += 1
                        continue
                self.stats.emitted += 1
                doc_stats.emitted += 1
                yield chunk

    def iter_document_stats(self) -> Iterator[tuple[SourceDocument, IngestionStats]]:
        for source_id, stats in self._document_stats.items():
            document = self._documents[source_id]
            yield document, stats

    # ------------------------------------------------------------------
    def _enumerate_paths(self, source: SourceConfig) -> List[Path]:
        base = source.path
        if not base.exists():
            return []
        candidates: set[Path] = set()
        globber = base.rglob if source.recursive else base.glob
        for pattern in source.include:
            for candidate in globber(pattern):
                if candidate.is_file():
                    candidates.add(candidate.resolve())
        filtered: List[Path] = []
        for candidate in sorted(candidates):
            if any(fnmatch(str(candidate), exclude) for exclude in source.exclude):
                continue
            filtered.append(candidate)
        return filtered

    def _resolve_handler(self, path: Path):
        if "faq" in path.stem.lower():
            try:
                handler_cls = self.registry.get("faq")
                return handler_cls()
            except KeyError:
                pass
        matches = self.registry.match(path)
        if matches:
            handler_name = matches[0]
            handler_cls = self.registry.get(handler_name)
            return handler_cls()

        default = self.config.ingest.default_handler
        if default:
            handler_cls = self.registry.get(default)
            return handler_cls()

        raise LookupError(
            f"No handler registered for {path.name}; update your config or register a plugin."
        )
