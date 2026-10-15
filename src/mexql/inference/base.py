"""Common interface of the generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from mexql.config import Config, ModelCfg, SamplingCfg

# A prompt is either a rendered string or an object with ``.text`` and optional ``.token_ids``
# (``mexql.prompt.PromptItem``). Backends prefer token ids when available.
PromptLike = Any


def prompt_text(p: PromptLike) -> str:
    return p if isinstance(p, str) else p.text


def prompt_ids(p: PromptLike) -> list[int] | None:
    return None if isinstance(p, str) else getattr(p, "token_ids", None)


@dataclass
class Completion:
    text: str
    finish_reason: str  # "stop" | "length" | other backend-specific values
    n_tokens: int


class Backend(Protocol):
    def generate(self, prompts: list[PromptLike], sampling: SamplingCfg, seed: int) -> list[list[Completion]]:
        """Return ``sampling.n`` completions for every prompt, in prompt order."""
        ...


class DummyBackend:
    """Deterministic stand-in for tests and pipeline smoke runs without a model."""

    DEFAULT = "<think>dummy backend</think>\n```sql\nSELECT 1\n```"

    def __init__(self, responder: Callable[[str], str] | None = None):
        self.responder = responder or (lambda _prompt: self.DEFAULT)

    def generate(self, prompts: list[PromptLike], sampling: SamplingCfg, seed: int) -> list[list[Completion]]:
        out = []
        for prompt in prompts:
            text = self.responder(prompt_text(prompt))
            out.append([Completion(text=text, finish_reason="stop", n_tokens=len(text.split())) for _ in range(sampling.n)])
        return out


def create_backend(cfg: Config, model_cfg: ModelCfg | None = None) -> Backend:
    name = cfg.backend
    if name == "dummy":
        return DummyBackend()
    model_cfg = model_cfg or cfg.require_model()
    if name == "vllm":
        from mexql.inference.vllm_backend import VllmBackend

        return VllmBackend(model_cfg, cfg.vllm, max_model_len=cfg.prompt.max_model_len, seed=cfg.seed)
    if name == "hf":
        from mexql.inference.hf_backend import HfBackend

        return HfBackend(model_cfg, cfg.hf, max_model_len=cfg.prompt.max_model_len)
    raise ValueError(f"Unknown backend {name!r}")
