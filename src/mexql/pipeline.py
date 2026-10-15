"""End-to-end steps shared by the evaluation and sampling scripts:
prompts -> generation -> parsing -> execution -> scoring -> JSONL records."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from mexql.config import Config, ModelCfg
from mexql.data.bird import BirdPaths, Example
from mexql.execution.cache import ResultCache, cache_key
from mexql.execution.pool import ExecTask, run_parallel
from mexql.execution.runner import ExecResult, results_match
from mexql.inference.base import Backend, Completion
from mexql.metrics import ScoredItem, final_status, summarize
from mexql.parsing import parse_response
from mexql.prompt import PromptItem, escaped_special_tags, make_prompt_item
from mexql.schema import load_or_extract_schema, render_ddl

log = logging.getLogger(__name__)


def load_tokenizer(model_cfg: ModelCfg):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_cfg.hf_id)


def gold_cache(cfg: Config) -> ResultCache:
    return ResultCache(Path(cfg.paths.processed_dir) / "cache" / "gold_results.sqlite")


def build_prompt_items(cfg: Config, examples: list[Example], tokenizer: Any, paths: BirdPaths,
                       chat_template_kwargs: dict | None = None) -> list[PromptItem]:
    ddl_by_db: dict[tuple[str, str], str] = {}
    items: list[PromptItem] = []
    escape = escaped_special_tags(tokenizer)
    if escape:
        log.info("escaping special tokens in prompts: %s", {k: v for k, v in escape.items()})
    for ex in examples:
        key = (ex.source, ex.db_id)
        if key not in ddl_by_db:
            schema = load_or_extract_schema(cfg, ex.source, ex.db_id, paths.db_path(ex.source, ex.db_id))
            ddl_by_db[key] = render_ddl(schema)
        items.append(make_prompt_item(ex, ddl_by_db[key], tokenizer, cfg.prompt.include_evidence,
                                      cfg.prompt.prompt_budget, chat_template_kwargs, escape))
    return items


def ensure_gold_results(cfg: Config, examples: Iterable[Example], paths: BirdPaths,
                        cache: ResultCache | None = None, retry_failed: bool = False) -> dict[str, ExecResult]:
    """Execute gold queries that are not cached yet; returns results keyed by example uid.

    With ``retry_failed`` cached timeouts / errors are executed again (useful after a run on a
    loaded or sleeping machine)."""
    cache = cache or gold_cache(cfg)
    examples = list(examples)
    keys = {ex.uid: cache_key(paths.db_rel_path(ex.source, ex.db_id), ex.gold_sql) for ex in examples}
    cached = cache.get_many(set(keys.values()))
    if retry_failed:
        cached = {k: v for k, v in cached.items() if v.ok}
    missing = {}
    for ex in examples:
        k = keys[ex.uid]
        if k not in cached and k not in missing:
            missing[k] = ExecTask(key=k, db_path=str(paths.db_path(ex.source, ex.db_id)), sql=ex.gold_sql)
    if missing:
        log.info("Executing %d gold queries (not cached)", len(missing))
        e = cfg.execution
        fresh = run_parallel(missing.values(), timeout_s=e.timeout_s, max_rows=e.max_rows,
                             num_workers=e.resolved_workers(), grace_s=e.worker_grace_s, desc="gold")
        cache.put_many(fresh.items())
        cached.update(fresh)
    return {ex.uid: cached[keys[ex.uid]] for ex in examples}


def execute_predictions(cfg: Config, examples_by_uid: dict[str, Example], parsed: list[tuple[str, int, str]],
                        paths: BirdPaths) -> dict[tuple[str, int], ExecResult]:
    """Execute parsed predictions (uid, sample_idx, sql); identical queries run once."""
    tasks: dict[str, ExecTask] = {}
    key_of: dict[tuple[str, int], str] = {}
    for uid, sample_idx, sql in parsed:
        ex = examples_by_uid[uid]
        k = cache_key(paths.db_rel_path(ex.source, ex.db_id), sql)
        key_of[(uid, sample_idx)] = k
        tasks.setdefault(k, ExecTask(key=k, db_path=str(paths.db_path(ex.source, ex.db_id)), sql=sql))
    e = cfg.execution
    results = run_parallel(tasks.values(), timeout_s=e.timeout_s, max_rows=e.max_rows,
                           num_workers=e.resolved_workers(), grace_s=e.worker_grace_s, desc="predictions")
    return {pair: results[k] for pair, k in key_of.items()}


def generate_and_score(cfg: Config, examples: list[Example], backend: Backend, tokenizer: Any, paths: BirdPaths,
                       cache: ResultCache | None = None, include_prompt: bool = True,
                       chat_template_kwargs: dict | None = None) -> tuple[list[dict], dict]:
    """Run the full pipeline; returns (records, timing) where records has one row per sample."""
    timing: dict[str, float] = {}
    t0 = time.monotonic()
    items = build_prompt_items(cfg, examples, tokenizer, paths, chat_template_kwargs)
    timing["prompts_s"] = time.monotonic() - t0
    n_long = sum(it.too_long for it in items)
    if n_long:
        log.warning("%d/%d prompts exceed the budget of %d tokens and will not be generated",
                    n_long, len(items), cfg.prompt.prompt_budget)

    t0 = time.monotonic()
    to_generate = [i for i, it in enumerate(items) if not it.too_long]
    generated = backend.generate([items[i] for i in to_generate], cfg.sampling, cfg.seed) if to_generate else []
    timing["generation_s"] = time.monotonic() - t0
    completions: list[list[Completion] | None] = [None] * len(items)
    for i, comps in zip(to_generate, generated):
        completions[i] = comps

    t0 = time.monotonic()
    gold = ensure_gold_results(cfg, examples, paths, cache)
    examples_by_uid = {ex.uid: ex for ex in examples}
    parsed_all: dict[tuple[str, int], Any] = {}
    to_execute: list[tuple[str, int, str]] = []
    for ex, comps in zip(examples, completions):
        for k, comp in enumerate(comps or []):
            p = parse_response(comp.text)
            parsed_all[(ex.uid, k)] = p
            if p.ok and gold[ex.uid].ok:
                to_execute.append((ex.uid, k, p.sql))
    exec_results = execute_predictions(cfg, examples_by_uid, to_execute, paths) if to_execute else {}
    timing["execution_s"] = time.monotonic() - t0

    records: list[dict] = []
    for ex, item, comps in zip(examples, items, completions):
        g = gold[ex.uid]
        samples = comps or [None] * cfg.sampling.n
        for k, comp in enumerate(samples):
            p = parsed_all.get((ex.uid, k))
            r = exec_results.get((ex.uid, k))
            match = results_match(r, g) if r is not None else False
            status = final_status(p.status if p else "format_error", r.status if r else None, g.ok, match,
                                  prompt_too_long=item.too_long)
            rec = {
                "uid": ex.uid, "source": ex.source, "question_id": ex.question_id, "db_id": ex.db_id,
                "difficulty": ex.difficulty, "sample_idx": k,
                "prompt_tokens": item.n_tokens, "prompt_too_long": item.too_long,
                "response": comp.text if comp else None,
                "finish_reason": comp.finish_reason if comp else None,
                "completion_tokens": comp.n_tokens if comp else None,
                "parse_status": p.status if p else None, "parse_reason": p.reason if p else None,
                "format_ok": p.format_ok if p else None, "format_issue": p.format_issue if p else None,
                "sql": p.sql if p else None,
                "exec_status": r.status if r else None, "exec_error": r.error if r else None,
                "n_rows": r.n_rows if r else None, "exec_elapsed_s": round(r.elapsed_s, 4) if r else None,
                "gold_status": g.status, "gold_error": g.error,
                "correct": (status == "correct") if status != "gold_invalid" else None,
                "status": status,
            }
            if include_prompt:
                rec["prompt"] = item.text
            records.append(rec)
    return records, timing


def summarize_records(records: list[dict]) -> dict:
    return summarize([ScoredItem(uid=r["uid"], difficulty=r.get("difficulty"), status=r["status"],
                                 format_ok=r.get("format_ok")) for r in records])


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked(items: list, size: int) -> Iterable[list]:
    """Split a list into consecutive chunks; size <= 0 means a single chunk."""
    if size <= 0 or size >= len(items):
        yield list(items)
        return
    for start in range(0, len(items), size):
        yield items[start:start + size]


def done_uids(path: str | Path) -> set[str]:
    """uids already present in a JSONL file (empty set if the file does not exist)."""
    path = Path(path)
    if not path.is_file():
        return set()
    return {row["uid"] for row in read_jsonl(path)}


def sync_outputs(out_dir: Path, sync_root: Path | None, outputs_root: Path, restore: bool = False) -> Path | None:
    """Mirror ``out_dir`` under ``sync_root`` (for example a mounted Google Drive) keeping the
    layout below ``outputs_root``. With ``restore`` the copy goes the other way: an existing
    mirror is copied back into an absent local ``out_dir`` so a run can resume in a fresh session."""
    if sync_root is None:
        return None
    import shutil

    try:
        rel = out_dir.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        rel = Path(out_dir.name)
    mirror = Path(sync_root) / rel
    if restore:
        if mirror.is_dir() and not out_dir.is_dir():
            shutil.copytree(mirror, out_dir)
            log.info("restored %s from %s", out_dir, mirror)
        return mirror
    if out_dir.is_dir():
        shutil.copytree(out_dir, mirror, dirs_exist_ok=True)
    return mirror
