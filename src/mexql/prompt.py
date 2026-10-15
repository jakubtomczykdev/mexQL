"""Prompt construction and token accounting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mexql.data.bird import Example

SYSTEM_PROMPT = """You are an expert SQLite analyst. Given a database schema, optional evidence and a question, write one SQLite query that answers the question.

Rules:
- First reason briefly (a few sentences) inside <think></think> tags.
- Then output exactly one SQL query inside a ```sql code block and nothing after it.
- Use only tables and columns that exist in the schema. Quote identifiers that contain spaces or special characters with backticks.
- Return exactly the columns the question asks for, nothing more.
- The query must be read-only (SELECT or WITH)."""

USER_TEMPLATE = """### Database schema (SQLite)
{ddl}

### Evidence
{evidence}

### Question
{question}

Reason inside <think></think>, then give the final query in a single ```sql code block."""


# Strings that appear literally in our instructions and are *special tokens* for some
# tokenizers (Qwen3). Fed as special tokens they push the model off-distribution (Qwen3-4B
# answers with a bare <tool_call>), so they are re-encoded as ordinary text.
ESCAPED_TAGS = ("<think>", "</think>")


@dataclass
class PromptItem:
    uid: str
    messages: list[dict[str, str]]
    text: str  # fully rendered prompt (chat template applied)
    n_tokens: int
    too_long: bool
    token_ids: list[int] | None = None  # prompt token ids with ESCAPED_TAGS encoded as plain text


def build_messages(example: Example, ddl: str, include_evidence: bool = True) -> list[dict[str, str]]:
    evidence = example.evidence if (include_evidence and example.evidence) else "(none)"
    user = USER_TEMPLATE.format(ddl=ddl, evidence=evidence, question=example.question)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], chat_template_kwargs: dict | None = None) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {})
    )


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


_TAG_PARTS = re.compile(r"(</?)([A-Za-z_]+)(>)")


def escaped_special_tags(tokenizer: Any) -> dict[int, list[int]]:
    """Map added-token id -> plain-text token ids for every tag in ESCAPED_TAGS that the
    tokenizer swallows as one added token (Qwen3 registers ``<think>`` and ``</think>`` as
    added tokens with ``special=False``, so ``split_special_tokens`` would not split them).
    Empty for tokenizers without such tokens (Qwen2.5)."""
    mapping: dict[int, list[int]] = {}
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tag in ESCAPED_TAGS:
        try:
            ids = tokenizer(tag, add_special_tokens=False)["input_ids"]
        except Exception:  # minimal tokenizers used in tests
            continue
        if len(ids) != 1 or ids[0] not in added:
            continue
        m = _TAG_PARTS.fullmatch(tag)
        if not m:
            continue
        plain: list[int] = []
        for part in m.groups():  # "<" / "</", name, ">" encoded separately cannot form the added token
            plain.extend(tokenizer(part, add_special_tokens=False)["input_ids"])
        if plain != ids and tokenizer.decode(plain) == tag:
            mapping[ids[0]] = plain
    return mapping


def encode_prompt(tokenizer: Any, text: str, escape: dict[int, list[int]] | None = None) -> list[int] | None:
    """Tokenize a rendered prompt; the chat template's own control tokens stay special, the
    tags listed in ESCAPED_TAGS become ordinary text. Returns None for tokenizers that do not
    produce integer ids (the whitespace stand-in)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids or not isinstance(ids[0], int):
        return None
    escape = escaped_special_tags(tokenizer) if escape is None else escape
    if not escape:
        return list(ids)
    out: list[int] = []
    for t in ids:
        out.extend(escape.get(t, (t,)))
    return out


def make_prompt_item(example: Example, ddl: str, tokenizer: Any, include_evidence: bool, budget: int,
                     chat_template_kwargs: dict | None = None, escape: dict[int, list[int]] | None = None) -> PromptItem:
    messages = build_messages(example, ddl, include_evidence)
    text = render_prompt(tokenizer, messages, chat_template_kwargs)
    token_ids = encode_prompt(tokenizer, text, escape)
    n = len(token_ids) if token_ids is not None else count_tokens(tokenizer, text)
    return PromptItem(uid=example.uid, messages=messages, text=text, n_tokens=n, too_long=n > budget,
                      token_ids=token_ids)


class WhitespaceTokenizer:
    """Minimal tokenizer stand-in for runs without a model (``--backend dummy``) and tests."""

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True, **_kwargs) -> str:
        return "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n<assistant>\n"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}
