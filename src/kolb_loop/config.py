from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class BackendConfig(BaseModel):
    base_url: str
    model: str
    api_key: str = "sk-no-key-required"


class BackendsConfig(BaseModel):
    main: BackendConfig
    reflector: BackendConfig
    embedder: BackendConfig


class EpisodicConfig(BaseModel):
    type: Literal["duckdb"] = "duckdb"
    path: str = "./data/episodic.duckdb"


class SemanticConfig(BaseModel):
    type: Literal["lancedb"] = "lancedb"
    path: str = "./data/semantic"


class MemoryConfig(BaseModel):
    episodic: EpisodicConfig = Field(default_factory=EpisodicConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)


class ReflectionTriggerConfig(BaseModel):
    trigger: Literal["sync", "async_after_each", "batch"] = "async_after_each"
    batch_interval_seconds: int = 300


class InjectionConfig(BaseModel):
    max_concepts_per_call: int = 3
    min_similarity: float = 0.6
    ab_test_ratio: float = 0.2


class LearningConfig(BaseModel):
    reflection: ReflectionTriggerConfig = Field(default_factory=ReflectionTriggerConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)


class IngressConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class SidecarConfig(BaseModel):
    ingress: IngressConfig = Field(default_factory=IngressConfig)
    fail_open: bool = True


class Settings(BaseSettings):
    sidecar: SidecarConfig = Field(default_factory=SidecarConfig)
    backends: BackendsConfig = Field(
        default_factory=lambda: BackendsConfig(
            main=BackendConfig(base_url="http://localhost:11434/v1", model="qwen3:8b"),
            reflector=BackendConfig(base_url="http://localhost:11434/v1", model="qwen3:8b"),
            embedder=BackendConfig(base_url="http://localhost:11434/v1", model="bge-m3"),
        )
    )
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    model_config = {"env_prefix": "KOLB_", "env_nested_delimiter": "__"}


def load_settings(config_path: Path | None = None) -> Settings:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text())
        return Settings.model_validate(raw)

    return Settings()
