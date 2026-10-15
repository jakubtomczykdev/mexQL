"""Plain transformers generation (CUDA, Apple MPS or CPU). Meant for smoke tests, not for full runs."""

from __future__ import annotations

import logging

from mexql.config import HfCfg, ModelCfg, SamplingCfg
from mexql.inference.base import Completion, PromptLike, prompt_ids, prompt_text

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HfBackend:
    def __init__(self, model_cfg: ModelCfg, hf_cfg: HfCfg, max_model_len: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(hf_cfg.device)
        self.batch_size = hf_cfg.batch_size
        self.max_model_len = max_model_len
        if model_cfg.dtype == "auto":
            if self.device == "cpu":
                dtype = torch.float32
            elif self.device == "cuda" and not torch.cuda.is_bf16_supported():
                dtype = torch.float16
            else:
                dtype = torch.bfloat16
        else:
            dtype = getattr(torch, model_cfg.dtype)
        log.info("Loading %s with transformers on %s (%s)", model_cfg.hf_id, self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = {t for t in eos + [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] if t is not None}

    def _finish(self, token_ids: list[int], max_tokens: int) -> tuple[list[int], str]:
        for i, t in enumerate(token_ids):
            if t in self.stop_ids:
                return token_ids[:i], "stop"
        return token_ids, "length" if len(token_ids) >= max_tokens else "stop"

    def _encode_batch(self, prompts: list[PromptLike]) -> dict:
        """Left-padded batch from precomputed token ids (preferred) or from text."""
        import torch

        ids_list = [prompt_ids(p) for p in prompts]
        if any(ids is None for ids in ids_list):
            return self.tokenizer([prompt_text(p) for p in prompts], return_tensors="pt", padding=True).to(self.device)
        width = max(len(ids) for ids in ids_list)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(ids_list), width), pad, dtype=torch.long)
        attention = torch.zeros((len(ids_list), width), dtype=torch.long)
        for row, ids in enumerate(ids_list):
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention[row, width - len(ids):] = 1
        return {"input_ids": input_ids.to(self.device), "attention_mask": attention.to(self.device)}

    def generate(self, prompts: list[PromptLike], sampling: SamplingCfg, seed: int) -> list[list[Completion]]:
        import torch
        from tqdm import tqdm

        torch.manual_seed(seed)
        results: list[list[Completion] | None] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompt_text(prompts[i])), reverse=True)
        gen_kwargs: dict = dict(max_new_tokens=sampling.max_tokens, num_return_sequences=sampling.n,
                                pad_token_id=self.tokenizer.pad_token_id)
        if sampling.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p, top_k=0)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        for start in tqdm(range(0, len(order), self.batch_size), desc="generating", unit="batch"):
            idxs = order[start:start + self.batch_size]
            # Prompts over the budget are filtered out by the pipeline, so no truncation here
            # (truncating would silently drop the question at the end of the prompt).
            enc = self._encode_batch([prompts[i] for i in idxs])
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1]:].tolist()
            for j, i in enumerate(idxs):
                comps = []
                for k in range(sampling.n):
                    ids, reason = self._finish(new_tokens[j * sampling.n + k], sampling.max_tokens)
                    # keep special tokens: for Qwen3 tokenizers <think> / </think> are special tokens and
                    # must survive decoding so the parser can see them (stop tokens are already cut off)
                    comps.append(Completion(text=self.tokenizer.decode(ids, skip_special_tokens=False),
                                            finish_reason=reason, n_tokens=len(ids)))
                results[i] = comps
        return results  # type: ignore[return-value]
