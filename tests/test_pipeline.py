import re

from mexql.config import load_config
from mexql.data.bird import BirdPaths, Example
from mexql.inference.base import DummyBackend
from mexql.pipeline import generate_and_score, read_jsonl, summarize_records, write_jsonl
from tests.conftest import REPO


class FakeTokenizer:
    """Whitespace tokenizer with a trivial chat template, so tests need no model download."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n<assistant>\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _examples():
    mk = lambda i, q, sql, diff: Example(f"dev:{i}", "dev", i, "tiny", q, "", sql, diff)
    return [
        mk(0, "How many people are there?", "SELECT COUNT(*) FROM people", "simple"),
        mk(1, "List the ids of people aged 30.", "SELECT id FROM people WHERE age = 30", "simple"),
        mk(2, "Total amount of orders of Ann Lee?", "SELECT SUM(o.amount) FROM orders o JOIN people p ON p.id = o.person_id WHERE p.`full name` = 'Ann Lee'", "moderate"),
        mk(3, "Broken gold.", "SELECT nope FROM people", "challenging"),
        mk(4, "Names of all people.", "SELECT `full name` FROM people", "moderate"),
        mk(5, "Timeout question.", "SELECT id FROM people", "simple"),
    ]


RESPONSES = {
    "How many people are there?": "<think>count</think>\n```sql\nSELECT COUNT(id) FROM people\n```",  # correct
    "List the ids of people aged 30.": "<think>x</think>\n```sql\nSELECT id FROM people WHERE age = 41\n```",  # wrong
    "Total amount of orders of Ann Lee?": "no sql at all",  # format error
    "Broken gold.": "<think>x</think>\n```sql\nSELECT 1\n```",  # gold invalid
    "Names of all people.": "<think>x</think>\n```sql\nSELECT nam FROM people\n```",  # sql error
    "Timeout question.": "<think>x</think>\n```sql\nWITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c\n```",
}


def _responder(prompt: str) -> str:
    m = re.search(r"### Question\n(.*?)\n\nReason", prompt, re.DOTALL)
    return RESPONSES[m.group(1).strip()]


def test_pipeline_end_to_end(tiny_raw_dir, tmp_path):
    cfg = load_config(
        [REPO / "configs" / "base.yaml"],
        overrides=[f"paths.raw_dir={tiny_raw_dir}", f"paths.processed_dir={tmp_path / 'p'}",
                   "execution.num_workers=2", "execution.timeout_s=0.5", "execution.worker_grace_s=1",
                   "backend=dummy", "prompt.max_model_len=4096", "prompt.max_new_tokens=256"],
    )
    paths = BirdPaths(cfg)
    examples = _examples()
    records, timing = generate_and_score(cfg, examples, DummyBackend(_responder), FakeTokenizer(), paths)
    assert len(records) == 6 and set(timing) == {"prompts_s", "generation_s", "execution_s"}
    by_q = {r["question_id"]: r for r in records}
    assert by_q[0]["status"] == "correct" and by_q[0]["correct"] is True
    assert by_q[1]["status"] == "wrong_result"
    assert by_q[2]["status"] == "format_error" and by_q[2]["parse_reason"] == "no_sql_block"
    assert by_q[0]["format_ok"] is True and by_q[2]["format_ok"] is False and by_q[2]["format_issue"] == "missing_think"
    assert by_q[3]["status"] == "gold_invalid" and by_q[3]["correct"] is None and by_q[3]["exec_status"] is None
    assert by_q[4]["status"] == "sql_error"
    assert by_q[5]["status"] == "timeout"
    assert "CREATE TABLE people" in by_q[0]["prompt"] and by_q[0]["prompt_tokens"] > 20
    s = summarize_records(records)
    assert s["n"] == 6 and s["n_scored"] == 5 and s["correct"] == 1 and s["ex"] == 20.0
    assert s["format_ok_rate"] == 80.0
    assert s["by_difficulty"]["simple"]["n"] == 3

    # gold results are cached: a second run executes no gold query (cache file exists and is reused)
    cache_file = tmp_path / "p" / "cache" / "gold_results.sqlite"
    assert cache_file.is_file()
    records2, _ = generate_and_score(cfg, examples, DummyBackend(_responder), FakeTokenizer(), paths,
                                     include_prompt=False)
    assert [r["status"] for r in records2] == [r["status"] for r in records]
    assert "prompt" not in records2[0]

    out = tmp_path / "pred.jsonl"
    write_jsonl(out, records)
    assert read_jsonl(out)[0]["uid"] == "dev:0"


def test_prompt_too_long(tiny_raw_dir, tmp_path):
    cfg = load_config(
        [REPO / "configs" / "base.yaml"],
        overrides=[f"paths.raw_dir={tiny_raw_dir}", f"paths.processed_dir={tmp_path / 'p'}",
                   "execution.num_workers=1", "backend=dummy", "prompt.max_model_len=64", "prompt.max_new_tokens=32"],
    )
    records, _ = generate_and_score(cfg, _examples()[:1], DummyBackend(), FakeTokenizer(), BirdPaths(cfg))
    assert records[0]["status"] == "prompt_too_long" and records[0]["response"] is None
