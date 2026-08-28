"""
Carica la configurazione da config/config.yaml e la espone come oggetti
tipizzati, cosi' il resto del progetto non tocca mai file yaml direttamente.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class LLMSettings:
    provider: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    model_name: str = "qwen/qwen3.6-35b-a3b"
    api_key: str = "lm-studio"
    temperature: float = 0.2
    max_tokens: int = 32678
    max_reasoning_tokens: int = 4092
    request_timeout_seconds: int = 500


@dataclass
class DatabaseSettings:
    provider: str = "sqlite"
    path: str = "data/recon_reports.db"

    def absolute_path(self) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class BB84Settings:
    qber_abort_threshold: float = 0.11


@dataclass
class AgentSettings:
    name: str = "recon_agent"
    max_llm_retries: int = 2


@dataclass
class PlanningSettings:
    min_hypotheses: int = 3
    max_hypotheses: int = 8
    max_llm_retries: int = 2


@dataclass
class ExecutionSettings:
    max_llm_retries: int = 2
    simulation_iterations: int = 5


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    bb84: BB84Settings = field(default_factory=BB84Settings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    planning: PlanningSettings = field(default_factory=PlanningSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Settings":
        if not path.exists():
            # Nessun file di config: usa i default, cosi' il progetto
            # funziona comunque "out of the box".
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        return cls(
            llm=LLMSettings(**raw.get("llm", {})),
            database=DatabaseSettings(**raw.get("database", {})),
            bb84=BB84Settings(**raw.get("bb84", {})),
            agent=AgentSettings(**raw.get("agent", {})),
            planning=PlanningSettings(**raw.get("planning", {})),
            execution=ExecutionSettings(**raw.get("execution", {})),
        )


_settings_singleton: Settings | None = None


def get_settings(force_reload: bool = False) -> Settings:
    """Restituisce le impostazioni globali (singleton), caricandole al primo uso."""
    global _settings_singleton
    if _settings_singleton is None or force_reload:
        _settings_singleton = Settings.from_yaml()
    return _settings_singleton
