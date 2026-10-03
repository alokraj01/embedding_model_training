"""Build NL2SQL prompts for RL from the table retriever's top-k tables.

For each question the retriever ranks tables, the top k are rendered as one-line CREATE TABLE
statements (original table/column names, so the SQL can run), and the prompt asks for one SQLite
query in a ```sql block. Prompts use *retrieved* tables, not gold ones, so training matches inference.

Pools (what the retriever ranks):
  --pool all : every table in tables.jsonl (876), as in `evaluate.py --pool all`. The top k nearly
               always spans several databases, so tables are grouped under "-- Database:" headers.
  --pool db  : only the question's own database (the database is assumed known at inference).

A question whose top k misses a gold table can never be answered. Those are dropped from train
(they would only waste rollouts) and kept with retrieval_complete=false in val/test, so end-to-end
accuracy counts them as failures. The complete share is the retriever's ceiling on accuracy.
The retriever was trained on train questions, so the train ceiling is optimistic.

Every gold query is also run on its Spider database (see sql_exec.py) and labelled with gold_status.
Train keeps only "ok" rows: a gold that errors can never be matched, and one that returns nothing,
only NULLs or a single 0 is matched by many wrong queries, which would reward them. Val/test keep
all rows with the label, so evaluation can decide what to exclude.

Outputs (data/rl/):
  {split}.jsonl  qid, db_id, question, gold_sql, gold_tables, retrieved_tables,
                 retrieval_complete, gold_status, prompt (chat messages: one user turn)
  stats.json

Usage:
  uv run python src/build_rl_prompts.py --model models/bge-small-mnrl-matryoshka/final --k 5
"""

import argparse
import functools
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from build_dataset import load_schemas, write_jsonl
from evaluate import BGE_QUERY_PROMPT, DATA, ROOT, read_jsonl
from sql_exec import ensure_databases, gold_status

OUT = ROOT / "data" / "rl"
SPLITS = ("train", "val", "test")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_PROBE = sqlite3.connect(":memory:")


@functools.cache
def quote(name):
    """Quote an identifier only if SQLite rejects it bare (spaces, or keywords such as `From`)."""
    if IDENT.fullmatch(name):
        try:
            _PROBE.execute(f"CREATE TEMP TABLE _probe ({name} TEXT)")
            _PROBE.execute("DROP TABLE _probe")
            return name
        except sqlite3.OperationalError:
            pass
    return '"' + name.replace('"', '""') + '"'


def parse_schema(db):
    """Per-table columns, primary keys and foreign keys, keyed by lowercase original table name."""
    names = db["table_names_original"]
    col_table = {ci: ti for ci, (ti, _) in enumerate(db["column_names_original"])}
    pks = set()
    for pk in db["primary_keys"]:
        pks.update(pk if isinstance(pk, list) else [pk])

    tables = {n.lower(): {"name": n, "cols": [], "pks": [], "fks": []} for n in names}
    for ci, (ti, col) in enumerate(db["column_names_original"]):
        if ti >= 0:
            t = tables[names[ti].lower()]
            t["cols"].append((col, db["column_types"][ci]))
            if ci in pks:
                t["pks"].append(col)
    for src, dst in db["foreign_keys"]:
        dst_table = names[col_table[dst]]
        tables[names[col_table[src]].lower()]["fks"].append(
            (db["column_names_original"][src][1], dst_table, db["column_names_original"][dst][1]))
    return tables


def create_table(t, shown):
    """One-line CREATE TABLE; foreign keys only to tables that are also in the prompt."""
    single_pk = t["pks"][0] if len(t["pks"]) == 1 else None
    parts = [f"{quote(c)} {typ}" + (" PRIMARY KEY" if c == single_pk else "") for c, typ in t["cols"]]
    if len(t["pks"]) > 1:
        parts.append(f"PRIMARY KEY ({', '.join(map(quote, t['pks']))})")
    parts += [f"FOREIGN KEY ({quote(src)}) REFERENCES {quote(dt)}({quote(dc)})"
              for src, dt, dc in t["fks"] if dt.lower() in shown]
    return f"CREATE TABLE {quote(t['name'])} ({', '.join(parts)});"


def render_schema(table_ids, schemas):
    """Group retrieved tables by database (databases ordered by their best-ranked table)."""
    by_db = defaultdict(list)
    for tid in table_ids:
        db_id, table = tid.split(".", 1)
        by_db[db_id].append(table)
    blocks = []
    for db_id, tables in by_db.items():
        shown = set(tables)
        lines = [f"-- Database: {db_id}"] + [create_table(schemas[db_id][t], shown) for t in tables]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(schema, question, multi_db):
    rule = " Use tables from one database only." if multi_db else ""
    return (f"{schema}\n\nQuestion: {question}\n"
            f"Write one SQLite query that answers the question.{rule} "
            f"Reply with only the SQL in a ```sql code block.")


def summarize(rows):
    n, ok = len(rows), sum(r["retrieval_complete"] for r in rows)
    chars = np.array([len(r["prompt"][0]["content"]) for r in rows])
    by_gold = defaultdict(lambda: [0, 0])
    for r in rows:
        g = str(min(len(r["gold_tables"]), 3)).replace("3", "3+")
        by_gold[g][0] += 1
        by_gold[g][1] += r["retrieval_complete"]
    return {
        "questions": n,
        "retrieval_complete": ok,
        "retrieval_missed": n - ok,
        "ceiling": round(ok / n, 4),
        "gold_status": dict(Counter(r["gold_status"] for r in rows)),
        "ceiling_by_gold_count": {g: {"n": a, "ceiling": round(b / a, 4)} for g, (a, b) in sorted(by_gold.items())},
        "avg_databases_in_prompt": round(float(np.mean([len({t.split(".")[0] for t in r["retrieved_tables"]})
                                                         for r in rows])), 2),
        "prompt_chars": {"mean": int(chars.mean()), "p50": int(np.percentile(chars, 50)),
                         "p95": int(np.percentile(chars, 95)), "max": int(chars.max())},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/bge-small-mnrl-matryoshka/final")
    p.add_argument("--k", type=int, default=5, help="tables put in each prompt")
    p.add_argument("--pool", default="all", choices=["all", "db"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    out = args.out.resolve()

    ensure_databases()
    schemas = {db_id: parse_schema(db) for db_id, db in load_schemas().items()}
    tables = read_jsonl(DATA / "tables.jsonl")
    table_ids = [t["table_id"] for t in tables]
    table_db = np.array([t["db_id"] for t in tables])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    t_emb = model.encode([t["text"] for t in tables], batch_size=args.batch_size,
                         normalize_embeddings=True, show_progress_bar=True)

    out.mkdir(parents=True, exist_ok=True)
    stats = {"model": args.model, "k": args.k, "pool": args.pool, "splits": {}}
    for split in SPLITS:
        queries = read_jsonl(DATA / f"{split}_queries.jsonl")
        q_emb = model.encode([BGE_QUERY_PROMPT + q["question"] for q in queries], batch_size=args.batch_size,
                             normalize_embeddings=True, show_progress_bar=True)
        sims = q_emb @ t_emb.T
        if args.pool == "db":
            sims[np.array([q["db_id"] for q in queries])[:, None] != table_db[None, :]] = -np.inf
        top = np.argsort(-sims, axis=1)[:, :args.k]

        rows = []
        for i, q in enumerate(queries):
            # with --pool db, a database smaller than k leaves -inf (other-database) slots in the top k
            retrieved = [table_ids[j] for j in top[i] if np.isfinite(sims[i, j])]
            schema = render_schema(retrieved, schemas)
            rows.append({
                "qid": q["qid"], "db_id": q["db_id"], "question": q["question"], "gold_sql": q["sql"],
                "gold_tables": q["gold_table_ids"], "retrieved_tables": retrieved,
                "retrieval_complete": set(q["gold_table_ids"]) <= set(retrieved),
                "gold_status": gold_status(q["db_id"], q["sql"]),
                "prompt": [{"role": "user", "content": build_prompt(schema, q["question"], args.pool == "all")}],
            })

        stats["splits"][split] = summarize(rows)
        if split == "train":
            rows = [r for r in rows if r["retrieval_complete"] and r["gold_status"] == "ok"]
        stats["splits"][split]["written"] = len(rows)
        write_jsonl(out / f"{split}.jsonl", rows)

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    for split, s in stats["splits"].items():
        print(f"{split:>5}: {s['questions']} questions, {s['retrieval_missed']} missed a gold table "
              f"-> ceiling {s['ceiling']:.1%}, gold {s['gold_status']}, wrote {s['written']} rows "
              f"(prompt chars p50={s['prompt_chars']['p50']}, p95={s['prompt_chars']['p95']})")
    print(f"saved {out}/")


if __name__ == "__main__":
    main()
