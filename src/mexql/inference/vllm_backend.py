"""Batched generation with vLLM (NVIDIA GPUs, Linux)."""

from __future__ import annotations

import logging

from mexql.config import ModelCfg, SamplingCfg, VllmCfg
from mexql.inference.base import Completion, PromptLike, prompt_ids, prompt_text

log = logging.getLogger(__name__)


def resolve_dtype(requested: str) -> str:
    """``auto`` -> bfloat16 where supported (Ampere+), float16 otherwise (e.g. Colab T4)."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


class VllmBackend:
    def __init__(self, model_cfg: ModelCfg, vllm_cfg: VllmCfg, max_model_len: int, seed: int):
        from vllm import LLM

        dtype = resolve_dtype(model_cfg.dtype)
        log.info("Loading %s with vLLM (dtype=%s, max_model_len=%d)", model_cfg.hf_id, dtype, max_model_len)
        self.llm = LLM(
            model=model_cfg.hf_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            enable_prefix_caching=vllm_cfg.enable_prefix_caching,
            max_num_seqs=vllm_cfg.max_num_seqs,
            seed=seed,
        )

    def generate(self, prompts: list[PromptLike], sampling: SamplingCfg, seed: int) -> list[list[Completion]]:
        from vllm import SamplingParams

        inputs = [{"prompt_token_ids": ids} if (ids := prompt_ids(p)) is not None else prompt_text(p) for p in prompts]

        params = SamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=seed,
            skip_special_tokens=False,  # keep <think> / </think>, which are special tokens for Qwen3
        )
        outputs = self.llm.generate(inputs, params, use_tqdm=True)
        return [
            [Completion(text=o.text, finish_reason=o.finish_reason or "unknown", n_tokens=len(o.token_ids)) for o in out.outputs]
            for out in outputs
        ]
