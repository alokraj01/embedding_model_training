"""Run SQL on the Spider SQLite databases and score it by execution match (the RL reward).

Databases live in data/raw/spider_db/spider_data/database/<db_id>/<db_id>.sqlite and are downloaded
once from a Hugging Face mirror of the Spider release (`python src/sql_exec.py`). The mirror was
checked against tables.json: every table and column is present in all 166 databases.

Execution match: the predicted query returns the same rows as the gold query. Row order counts only
when the gold query has a top-level ORDER BY; otherwise rows are compared as multisets. Columns may
come back in a different order (as in Spider's official execution evaluation). Floats are rounded
to 6 places. Databases are opened read-only, so model-written SQL cannot modify them.

Gold status (whether a gold query is usable as a reward target):
  ok         returns at least one row with a non-NULL value
  empty      returns no rows: any query that returns nothing would match it
  null_only  every value is NULL (e.g. MAX over an empty table): same problem
  zero       a single 0 (e.g. a count that matches nothing): any over-filtered count would match it
  error      fails to run (broken Spider annotation)
  timeout    runs longer than the time limit
"""

import functools
import itertools
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parent.parent
DB_ROOT = ROOT / "data" / "raw" / "spider_db"
DB_DIR = DB_ROOT / "spider_data" / "database"
DB_MIRROR = "minktn/spider-data"  # HF dataset repo with spider_data/database/<db_id>/<db_id>.sqlite

TIMEOUT_S = 2.0  # slowest gold query takes 0.4 s
MAX_ROWS = 10_000
MAX_PERMUTED_COLS = 6  # try column permutations only for results this narrow
SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.S | re.I)


def ensure_databases():
    """Download the Spider databases if they are not there yet."""
    if not DB_DIR.exists():
        from huggingface_hub import snapshot_download

        snapshot_download(DB_MIRROR, repo_type="dataset", local_dir=DB_ROOT,
                          allow_patterns=["spider_data/database/*/*.sqlite"])
    return DB_DIR


@functools.cache
def _connect(db_id):
    path = DB_DIR / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run `python src/sql_exec.py` to download the databases)")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="replace")  # a few Spider DBs hold non-UTF-8 bytes
    return conn


def execute(db_id, sql, timeout=TIMEOUT_S):
    """Return (rows, None) or (None, error message). Error is "timeout" when the time limit hits."""
    conn = _connect(db_id)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS)
        return [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows], None
    except Exception as e:  # model output can fail in many ways (syntax, two statements, overflow...)
        return None, "timeout" if time.monotonic() > deadline else f"{type(e).__name__}: {e}"
    finally:
        conn.set_progress_handler(None, 0)


@functools.lru_cache(maxsize=None)
def gold_result(db_id, sql):
    """Gold rows (cached, since each gold query is scored against many rollouts) and whether order counts."""
    rows, err = execute(db_id, sql)
    return rows, err, is_ordered(sql)


def is_ordered(sql):
    try:
        return bool(sqlglot.parse_one(sql, read="sqlite").args.get("order"))
    except sqlglot.errors.ParseError:
        return "order by" in sql.lower()


def gold_status(db_id, sql):
    rows, err, _ = gold_result(db_id, sql)
    if err:
        return "timeout" if err == "timeout" else "error"
    if not rows:
        return "empty"
    if all(v is None for r in rows for v in r):
        return "null_only"
    if rows == [(0,)]:
        return "zero"
    return "ok"


def same_result(pred, gold, ordered):
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    n = len(gold[0])
    if len(pred[0]) != n:
        return False
    identity = tuple(range(n))
    perms = itertools.permutations(range(n)) if n <= MAX_PERMUTED_COLS else [identity]
    gold_cols = [Counter(r[i] for r in gold) for i in range(n)]
    pred_cols = [Counter(r[j] for r in pred) for j in range(n)]
    fits = [[pred_cols[j] == gold_cols[i] for j in range(n)] for i in range(n)]  # pred col j can be gold col i
    for p in sorted(perms, key=lambda p: p != identity):  # identity first
        if not all(fits[i][p[i]] for i in range(n)):
            continue
        rows = [tuple(r[j] for j in p) for r in pred]
        if (rows == gold) if ordered else (Counter(rows) == Counter(gold)):
            return True
    return False


def execution_match(db_id, pred_sql, gold_sql):
    if not pred_sql:
        return False
    gold, err, ordered = gold_result(db_id, gold_sql)
    if err:
        return False
    pred, err = execute(db_id, pred_sql)
    return err is None and same_result(pred, gold, ordered)


def extract_sql(text):
    """SQL from the last ```sql block in a completion, or None if there is no code block."""
    blocks = SQL_BLOCK.findall(text)
    return blocks[-1].strip() if blocks and blocks[-1].strip() else None


def execution_reward(completions, db_id, gold_sql, **kwargs):
    """TRL-style reward: 1.0 if the completion's SQL returns the gold result, else 0.0.

    TRL passes the dataset's extra columns (db_id, gold_sql) as keyword lists aligned with completions.
    """
    texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
    return [float(execution_match(d, extract_sql(t), g)) for t, d, g in zip(texts, db_id, gold_sql)]


def main():
    """Download the databases and check them against tables.json."""
    ensure_databases()
    schemas = json.loads((ROOT / "data" / "raw" / "tables.json").read_text())
    problems = []
    for db in schemas:
        conn = _connect(db["db_id"])
        real = {t.lower(): {r[1].lower() for r in conn.execute(f'PRAGMA table_info("{t}")')}
                for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for ti, col in db["column_names_original"]:
            t = db["table_names_original"][ti].lower() if ti >= 0 else None
            if t is not None and col.lower() not in real.get(t, set()):
                problems.append(f"{db['db_id']}.{t}.{col}")
    print(f"{len(schemas)} databases in {DB_DIR.relative_to(ROOT)}; "
          f"columns in tables.json missing from the databases: {len(problems)} {problems[:10]}")


if __name__ == "__main__":
    main()
