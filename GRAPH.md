# Graphiti Integration Notes

Graphiti is LlamaIndex's graph-based knowledge stack. It takes unstructured documents, extracts entities and relationships, stores them in a graph database, and exposes GraphRAG retrieval. Instead of relying solely on vector similarity, it enables reasoning over graph structure (paths, neighborhoods, subgraphs) and blending those results with chunk retrieval.

## Potential Integration Outline

### Ingestion
- Run a parallel pipeline after producing clean text chunks to invoke Graphiti's extractor (e.g., LlamaIndex `KnowledgeGraphIndex` or Graphiti REST API).
- Batch chunks into Graphiti's node/edge format, carrying source document IDs for traceability.
- Handle per-tenant credentials and reuse content hashes to avoid reprocessing unchanged chunks.

### Storage & Retrieval
- Introduce an optional `GraphitiStore` (or similar) that encapsulates Graphiti's API and pagination.
- Retrieval could fan out: query Graphiti for relevant subgraphs, then merge/rerank with pgvector hits to keep hybrid answers.
- Config would toggle graph support, capturing connection details, graph traversal parameters (depth, entity filters), and credential injection.

### Operational Considerations
- Graphiti depends on its own Postgres + Redis stack; document infra requirements and how per-tenant API keys map through the CLI flags we already support.
- Graph extraction is slower than chunking, so expose a CLI flag (`--graph`) or config section to opt in rather than forcing all ingestions to run it.
- Testing strategy: fixture-based tiny graphs plus HTTP mocks, since Graphiti is exposed as an API/service.

## Recommendation
Adding Graphiti makes sense if you need relationship-heavy answers (ownership, dependency chains) or want GraphRAG experimentation. Otherwise it's extra complexity (new infra, more moving parts). We can plan for it by keeping ingestion hooks and storage abstractions pluggable, then ship Graphiti as an opt-in plugin when demand appears.
