import pytest

from mexql.parsing import parse_response

GOOD = "<think>Join people and orders.</think>\n```sql\nSELECT id FROM people;\n```"


def test_good_response():
    p = parse_response(GOOD)
    assert p.ok and p.sql == "SELECT id FROM people" and p.think == "Join people and orders."
    assert p.format_ok and p.format_issue is None and p.n_sql_blocks == 1


@pytest.mark.parametrize(
    "text, reason",
    [
        ("", "empty_response"),
        (None, "empty_response"),
        ("<think>thinking forever", "unterminated_think"),
        ("<think>ok</think> no code here", "no_sql_block"),
        ("<think>ok</think>\n```sql\n\n```", "empty_sql"),
        ("<think>ok</think>\n```sql\nDROP TABLE people\n```", "not_a_select"),
        ("```python\nprint(1)\n```", "no_sql_block"),
    ],
)
def test_format_errors(text, reason):
    p = parse_response(text)
    assert p.status == "format_error" and p.sql is None and p.reason == reason and not p.format_ok


def test_last_sql_block_wins_and_blocks_inside_think_are_ignored():
    text = "<think>```sql\nSELECT 0\n```</think>\n```sql\nSELECT 1\n```\ntext\n```sql\nSELECT 2\n```"
    p = parse_response(text)
    assert p.ok and p.sql == "SELECT 2" and p.n_sql_blocks == 2
    assert not p.format_ok and p.format_issue == "multiple_sql_blocks"


def test_lenient_extraction_with_format_issues():
    # think never closed, SQL at the end: SQL is used, format is flagged
    p = parse_response("<think>\nstep 1... final query:\n```sql\nSELECT a FROM t\n```")
    assert p.ok and p.sql == "SELECT a FROM t" and p.think is None
    assert not p.format_ok and p.format_issue == "unterminated_think"
    # no think at all
    p = parse_response("```sql\nSELECT a FROM t\n```")
    assert p.ok and not p.format_ok and p.format_issue == "missing_think"
    # untagged fence holding a query
    p = parse_response("<think>x</think>\nHere you go:\n```\nSELECT a FROM t\n```")
    assert p.ok and p.sql == "SELECT a FROM t" and p.format_issue == "untagged_sql_block"
    # text before the think block
    p = parse_response("Sure!\n<think>x</think>\n```sql\nSELECT 1\n```")
    assert p.ok and p.format_issue == "text_before_think"


def test_language_tag_case_and_with_cte():
    p = parse_response("<think>x</think>\n```SQL\nWITH c AS (SELECT 1 AS x) SELECT x FROM c;;\n```")
    assert p.ok and p.format_ok and p.sql == "WITH c AS (SELECT 1 AS x) SELECT x FROM c"


def test_multiline_sql_is_preserved():
    p = parse_response("<think>x</think>\n```sql\nSELECT a,\n       b\nFROM t\nWHERE a > 1\n```")
    assert p.ok and p.sql == "SELECT a,\n       b\nFROM t\nWHERE a > 1"
