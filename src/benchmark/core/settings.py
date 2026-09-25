from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from benchmark.infra.guard import exigir_alvo_permitido


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    benchmark_env: str = "local"
    # Portas deste repositório. NÃO repor 5432, 5433 ou a banda 17xxx:
    # pertencem ao ambiente experimental original, e o travão em
    # benchmark.infra.guard recusa-as ao carregar as definições.
    database_url: str = "postgresql://benchmark:benchmark@localhost:15432/benchmark"
    graphrag_neo4j_uri: str = "bolt://localhost:18687"
    graphrag_neo4j_user: str = "neo4j"
    graphrag_neo4j_password: str = "benchmark_graphrag"
    graphrag_neo4j_database: str = "neo4j"
    lightrag_neo4j_uri: str = "bolt://localhost:18688"
    lightrag_neo4j_user: str = "neo4j"
    lightrag_neo4j_password: str = "benchmark_lightrag"
    lightrag_neo4j_database: str = "neo4j"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    config_dir: Path = Path("configs")
    model_provider: str = "fake"
    openai_api_key: str | None = None
    openai_chat_model: str | None = None
    # Responses API does not expose a seed parameter (openai 2.41.0), so
    # temperature=0 is the only determinism knob available to the reader.
    openai_chat_temperature: float = 0.0
    openai_embedding_model: str | None = None
    openai_embedding_dimensions: int | None = None

    @model_validator(mode="after")
    def _recusar_ambiente_original(self) -> Settings:
        """Travão: nenhuma definição pode apontar ao ambiente da dissertação.

        Corre em toda a carga de definições, portanto apanha tanto os valores
        por omissão como um `.env` mal preenchido.
        """
        exigir_alvo_permitido(self.database_url, origem="DATABASE_URL")
        exigir_alvo_permitido(self.graphrag_neo4j_uri, origem="GRAPHRAG_NEO4J_URI")
        exigir_alvo_permitido(self.lightrag_neo4j_uri, origem="LIGHTRAG_NEO4J_URI")
        return self


def load_settings() -> Settings:
    return Settings()
