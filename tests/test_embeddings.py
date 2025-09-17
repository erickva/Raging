from raging.config import EmbeddingConfig
from raging.embeddings.proxy import OpenAIProxyEmbeddingClient


def test_openai_proxy_embedding_accepts_inline_api_key(monkeypatch):
    config = EmbeddingConfig(
        model="text-embedding-3-small",
        base_url="https://proxy.local/v1",
        api_key="tenant-key",
    )
    client = OpenAIProxyEmbeddingClient(config)
    assert client.session.headers["Authorization"] == "Bearer tenant-key"


def test_openai_proxy_embedding_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("TENANT_KEY", "env-secret")
    config = EmbeddingConfig(
        model="text-embedding-3-small",
        base_url="https://proxy.local/v1",
        api_key_env="TENANT_KEY",
    )
    client = OpenAIProxyEmbeddingClient(config)
    assert client.session.headers["Authorization"] == "Bearer env-secret"
