from mexql.metrics import ScoredItem, final_status, format_markdown, format_table, summarize


def test_final_status():
    assert final_status("ok", "ok", True, True) == "correct"
    assert final_status("ok", "ok", True, False) == "wrong_result"
    assert final_status("format_error", None, True, False) == "format_error"
    assert final_status("ok", "timeout", True, False) == "timeout"
    assert final_status("ok", "too_large", True, False) == "too_large"
    assert final_status("ok", "sql_error", True, False) == "sql_error"
    assert final_status("ok", "ok", False, False) == "gold_invalid"
    assert final_status("ok", None, True, False, prompt_too_long=True) == "prompt_too_long"


def test_summarize_by_difficulty():
    items = [
        ScoredItem("a", "simple", "correct", format_ok=True),
        ScoredItem("b", "simple", "wrong_result", format_ok=False),
        ScoredItem("c", "moderate", "correct"),
        ScoredItem("d", "challenging", "sql_error"),
        ScoredItem("e", "challenging", "gold_invalid"),
        ScoredItem("f", None, "timeout"),
    ]
    s = summarize(items)
    assert s["n"] == 6 and s["n_scored"] == 5 and s["correct"] == 2
    assert s["ex"] == 40.0 and abs(s["ex_all"] - 100 * 2 / 6) < 1e-9
    assert s["by_difficulty"]["simple"] == {"n": 2, "correct": 1, "ex": 50.0}
    assert s["by_difficulty"]["challenging"]["n"] == 1
    assert s["by_difficulty"]["unlabeled"]["n"] == 1
    assert s["status_counts"] == {"correct": 2, "wrong_result": 1, "sql_error": 1, "timeout": 1, "gold_invalid": 1}
    assert s["format_ok_rate"] == 50.0  # only items with a known format flag count
    table = format_table(s, "demo")
    assert "simple" in table and "40.00" in table and "gold_invalid=1" in table
    md = format_markdown(s, "model/dev")
    assert md.startswith("| model / split |") and "| model/dev |" in md


def test_summarize_empty():
    s = summarize([])
    assert s["n"] == 0 and s["ex"] == 0.0 and s["by_difficulty"] == {} and s["format_ok_rate"] is None
