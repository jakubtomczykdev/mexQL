"""YAML configuration with dotted command-line overrides and pydantic validation.

Usage pattern in scripts::

    cfg = load_config(["configs/base.yaml", "configs/evaluate.yaml", "configs/models/qwen3_4b_instruct.yaml"],
                      overrides=["split=val_internal", "sampling.temperature=0.2"])
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PathsCfg(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")


class DataCfg(BaseModel):
    dev_dirname: str = "dev_20240627"
    train_dirname: str = "train"
    val_internal_size: int = 500


class PromptCfg(BaseModel):
    max_examples_per_column: int = 3
    max_example_chars: int = 40
    example_scan_rows: int = 200
    include_evidence: bool = True
    max_model_len: int = 16384
    max_new_tokens: int = 1024

    @property
    def prompt_budget(self) -> int:
        """Maximum prompt length in tokens that still leaves room for generation."""
        return self.max_model_len - self.max_new_tokens


class ExecutionCfg(BaseModel):
    timeout_s: float = 30.0
    max_rows: int = 500_000
    num_workers: int | Literal["auto"] = "auto"
    worker_grace_s: float = 5.0

    def resolved_workers(self) -> int:
        if self.num_workers == "auto":
            return max(1, min(os.cpu_count() or 1, 8))
        return max(1, int(self.num_workers))


class ModelCfg(BaseModel):
    name: str
    hf_id: str
    dtype: str = "auto"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class SamplingCfg(BaseModel):
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024


class VllmCfg(BaseModel):
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    max_num_seqs: int = 64


class HfCfg(BaseModel):
    batch_size: int = 4
    device: str = "auto"


class Config(BaseModel):
    seed: int = 1234
    paths: PathsCfg = Field(default_factory=PathsCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    prompt: PromptCfg = Field(default_factory=PromptCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    model: ModelCfg | None = None
    backend: Literal["vllm", "hf", "dummy"] = "vllm"
    split: str = "dev"
    limit: int | None = None
    smoke_limit: int = 20
    run_name: str | None = None
    sampling: SamplingCfg = Field(default_factory=SamplingCfg)
    vllm: VllmCfg = Field(default_factory=VllmCfg)
    hf: HfCfg = Field(default_factory=HfCfg)

    def require_model(self) -> ModelCfg:
        if self.model is None:
            raise ValueError("No model configured: pass --model configs/models/<name>.yaml")
        return self.model


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base`` (dicts merge, other values replace)."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict[str, Any], override: str) -> None:
    """Apply one ``a.b.c=value`` override in place; the value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw) if raw != "" else None
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = value


def load_config(files: list[str | Path], overrides: list[str] | tuple[str, ...] = ()) -> Config:
    """Load and merge YAML files in order, then apply overrides and validate."""
    merged: dict[str, Any] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{file}: top level must be a mapping")
        merged = deep_merge(merged, data)
    for override in overrides:
        apply_override(merged, override)
    return Config.model_validate(merged)
