import pytest

from mexql.data.bird import Example
from mexql.prompt import ESCAPED_TAGS, WhitespaceTokenizer, build_messages, encode_prompt, escaped_special_tags, make_prompt_item


def _tokenizer(name):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"tokenizer {name} not cached locally")


def test_build_messages_contains_schema_evidence_question():
    ex = Example("dev:1", "dev", 1, "db", "How many?", "count means COUNT", "SELECT 1", "simple")
    msgs = build_messages(ex, "CREATE TABLE t (a INT);")
    assert msgs[0]["role"] == "system" and "<think></think>" in msgs[0]["content"]
    assert "CREATE TABLE t (a INT);" in msgs[1]["content"] and "count means COUNT" in msgs[1]["content"]
    assert build_messages(ex, "x", include_evidence=False)[1]["content"].count("(none)") == 1


def test_whitespace_tokenizer_gives_no_ids():
    tok = WhitespaceTokenizer()
    ex = Example("dev:1", "dev", 1, "db", "q", "", "SELECT 1", None)
    item = make_prompt_item(ex, "ddl", tok, True, budget=5)
    assert item.token_ids is None and item.too_long and item.n_tokens > 5
    assert escaped_special_tags(tok) == {}


def test_qwen3_think_tags_are_escaped_to_plain_text():
    tok = _tokenizer("Qwen/Qwen3-4B-Instruct-2507")
    escape = escaped_special_tags(tok)
    assert len(escape) == 2  # <think> and </think> are single added tokens for Qwen3
    for tag in ESCAPED_TAGS:
        (sid,) = tok(tag, add_special_tokens=False)["input_ids"]
        assert sid in escape and len(escape[sid]) > 1 and tok.decode(escape[sid]) == tag
    text = tok.apply_chat_template(build_messages(
        Example("dev:1", "dev", 1, "db", "q", "", "SELECT 1", None), "ddl"), tokenize=False, add_generation_prompt=True)
    ids = encode_prompt(tok, text, escape)
    assert not (set(ids) & set(escape))  # no special think tokens left in the prompt
    assert tok.decode(ids) == text  # text is unchanged, only the tokenization differs
    assert tok("<|im_start|>", add_special_tokens=False)["input_ids"][0] in ids  # template tokens stay special


def test_qwen25_has_no_special_think_tokens():
    tok = _tokenizer("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    assert escaped_special_tags(tok) == {}
    text = "<|im_start|>user\n<think></think><|im_end|>\n"
    assert encode_prompt(tok, text) == tok(text, add_special_tokens=False)["input_ids"]
