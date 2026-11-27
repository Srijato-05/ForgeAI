"""Configuration, provider endpoints, model roles, and budget settings for ForgeAI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["groq", "openrouter", "sambanova", "ollama"]
ModelRole = Literal["factory", "agent", "judge"]

PROVIDER_BASE_URLS: dict[ProviderName, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "ollama": "http://localhost:11434/v1",
}

DEFAULT_MODELS: dict[ProviderName, dict[ModelRole, str]] = {
    "groq": {
        "factory": "qwen/qwen3.8-27b",
        "agent": "openai/gpt-oss-20b",
        "judge": "qwen/qwen3.8-27b",
    },
    "openrouter": {
        "factory": "qwen/qwen3-8b:free",
        "agent": "meta-llama/llama-3.1-8b-instruct:free",
        "judge": "google/gemma-3-12b-it:free",
    },
    "sambanova": {
        "factory": "Qwen2.5-32B-Instruct",
        "agent": "Meta-Llama-3.1-8B-Instruct",
        "judge": "Qwen2.5-32B-Instruct",
    },
    "ollama": {
        "factory": "qwen2.5:3b",
        "agent": "llama3.2:3b",
        "judge": "gemma3:4b",
    },
}


class BudgetConfig(BaseModel):
    """Execution budget and stopping thresholds for the critic-optimizer loop."""

    max_tokens: int = 250_000
    max_latency_seconds: float = 600.0
    max_iterations: int = 3
    pass_threshold: float = 0.90
    min_marginal_gain: float = 0.03


class ForgeSettings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    forge_provider: ProviderName = Field(default="groq", alias="FORGE_PROVIDER")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    sambanova_api_key: str = Field(default="", alias="SAMBANOVA_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")

    forge_factory_model: str = Field(default="", alias="FORGE_FACTORY_MODEL")
    forge_agent_model: str = Field(default="", alias="FORGE_AGENT_MODEL")
    forge_judge_model: str = Field(default="", alias="FORGE_JUDGE_MODEL")

    cache_dir: Path = Field(default=Path(".forge_cache"))
    runs_dir: Path = Field(default=Path("runs"))
    rpm_limit: int = Field(default=25)
    tpm_limit: int = Field(default=10_000)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    def get_base_url(self, provider: ProviderName | None = None) -> str:
        """Return the OpenAI-compatible base URL for the active or given provider."""
        target = provider or self.forge_provider
        if target == "ollama":
            return self.ollama_base_url
        return PROVIDER_BASE_URLS[target]

    def get_model(self, role: ModelRole, provider: ProviderName | None = None) -> str:
        """Return the configured model name for a given role, respecting env overrides."""
        override_map: dict[ModelRole, str] = {
            "factory": self.forge_factory_model,
            "agent": self.forge_agent_model,
            "judge": self.forge_judge_model,
        }
        override = override_map[role].strip()
        if override:
            return override
        target = provider or self.forge_provider
        return DEFAULT_MODELS[target][role]

    def get_api_key(self, provider: ProviderName | None = None) -> str:
        """Return the API key for the requested provider (non-empty placeholder for Ollama)."""
        target = provider or self.forge_provider
        if target == "groq":
            return self.groq_api_key
        if target == "openrouter":
            return self.openrouter_api_key
        if target == "sambanova":
            return self.sambanova_api_key
        return "ollama"

