"""End-to-end runs of the two entry points with the dummy backend on a tiny BIRD-like raw dir:
chunking, resume (no duplicates, nothing regenerated) and mirroring to a sync directory."""

import json
import sys
from pathlib import Path

from mexql.pipeline import read_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate  # noqa: E402
import sample_train  # noqa: E402

QUESTIONS = [
    {"question_id": 0, "db_id": "tiny", "question": "How many people?", "evidence": "", "SQL": "SELECT COUNT(*) FROM people", "difficulty": "simple"},
    {"question_id": 1, "db_id": "tiny", "question": "Ids of people aged 30.", "evidence": "", "SQL": "SELECT id FROM people WHERE age = 30", "difficulty": "simple"},
    {"question_id": 2, "db_id": "tiny", "question": "Names.", "evidence": "", "SQL": "SELECT `full name` FROM people", "difficulty": "moderate"},
    {"question_id": 3, "db_id": "tiny", "question": "Orders of Ann.", "evidence": "", "SQL": "SELECT order_id FROM orders WHERE person_id = 1", "difficulty": "challenging"},
    {"question_id": 4, "db_id": "tiny", "question": "Max height.", "evidence": "", "SQL": "SELECT MAX(height) FROM people", "difficulty": "simple"},
]


def _args(tiny_raw_dir, tmp_path, extra):
    return [
        "--backend", "dummy", "--split", "dev",
        "--set", f"paths.raw_dir={tiny_raw_dir}", "--set", f"paths.processed_dir={tmp_path / 'processed'}",
        "--set", f"paths.outputs_dir={tmp_path / 'outputs'}", "--set", "execution.num_workers=1",
        "--set", "prompt.max_model_len=4096", "--set", "prompt.max_new_tokens=256",
    ] + extra


def _prepare(tiny_raw_dir):
    (tiny_raw_dir / "dev_20240627" / "dev.json").write_text(json.dumps(QUESTIONS))


def test_evaluate_chunked_resume_and_sync(tiny_raw_dir, tmp_path):
    _prepare(tiny_raw_dir)
    sync = tmp_path / "drive"
    run = tmp_path / "outputs" / "runs" / "dummy_dev"
    assert evaluate.main(_args(tiny_raw_dir, tmp_path, ["--chunk-size", "2", "--sync-dir", str(sync)])) == 0
    preds = read_jsonl(run / "predictions.jsonl")
    assert [r["question_id"] for r in preds] == [0, 1, 2, 3, 4]
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["n_questions"] == 5 and metrics["n_generated_this_session"] == 5
    assert (sync / "runs" / "dummy_dev" / "predictions.jsonl").is_file()  # mirrored

    # resume with nothing left to do: no duplicates, nothing generated
    assert evaluate.main(_args(tiny_raw_dir, tmp_path, ["--resume", "--chunk-size", "2"])) == 0
    assert len(read_jsonl(run / "predictions.jsonl")) == 5
    assert json.loads((run / "metrics.json").read_text())["n_generated_this_session"] == 0

    # partial file + resume: only the missing questions are generated
    kept = preds[:3]
    (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    assert evaluate.main(_args(tiny_raw_dir, tmp_path, ["--resume"])) == 0
    again = read_jsonl(run / "predictions.jsonl")
    assert [r["question_id"] for r in again] == [0, 1, 2, 3, 4]
    assert json.loads((run / "metrics.json").read_text())["n_generated_this_session"] == 2

    # fresh local directory + resume restores the mirror from the sync dir
    import shutil

    shutil.rmtree(run)
    assert evaluate.main(_args(tiny_raw_dir, tmp_path, ["--resume", "--sync-dir", str(sync)])) == 0
    assert json.loads((run / "metrics.json").read_text())["n_generated_this_session"] == 0
    assert len(read_jsonl(run / "predictions.jsonl")) == 5

    # without --resume the run starts over
    assert evaluate.main(_args(tiny_raw_dir, tmp_path, [])) == 0
    assert len(read_jsonl(run / "predictions.jsonl")) == 5


def test_sample_train_chunked_resume(tiny_raw_dir, tmp_path):
    _prepare(tiny_raw_dir)
    run = tmp_path / "outputs" / "sampling" / "dummy_dev_n2"
    extra = ["--set", "sampling.n=2", "--limit", "0", "--chunk-size", "2"]
    assert sample_train.main(_args(tiny_raw_dir, tmp_path, extra)) == 0
    rates = read_jsonl(run / "pass_rates.jsonl")
    assert [r["question_id"] for r in rates] == [0, 1, 2, 3, 4] and all(r["n"] == 2 for r in rates)
    assert len(read_jsonl(run / "samples.jsonl")) == 10
    summary = json.loads((run / "summary.json").read_text())
    assert summary["n_questions"] == 5 and summary["estimate_full_split"]["based_on_questions"] == 5
    assert sum(summary["n_correct_histogram"].values()) == 5
    assert sum(summary["sample_status_counts"].values()) == 10

    assert sample_train.main(_args(tiny_raw_dir, tmp_path, extra + ["--resume"])) == 0
    assert len(read_jsonl(run / "pass_rates.jsonl")) == 5 and len(read_jsonl(run / "samples.jsonl")) == 10
    summary = json.loads((run / "summary.json").read_text())
    assert summary["n_generated_this_session"] == 0 and summary["estimate_full_split"]["total_hours"] is None
