"""Build (question, table description) training pairs with same-database hard negatives from Spider.

Splits are by database, so val/test databases are never seen in training:
  train / val : Spider train databases, split by db_id (VAL_FRACTION of DBs held out)
  test        : Spider dev databases (disjoint from Spider train)

Outputs (data/processed/):
  tables.jsonl              every table description: {table_id, db_id, table, text}
  {split}_pairs.jsonl       one row per (question, gold table): anchor, positive, hard_negatives, ...
  train_triplets.jsonl      (anchor, positive, negative) rows for MultipleNegativesRankingLoss
  {split}_queries.jsonl     one row per question with all gold table ids (for recall@k eval)
  stats.json
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import sqlglot
from datasets import load_dataset
from sqlglot import exp

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
TABLES_URL = "https://raw.githubusercontent.com/taoyds/spider/master/evaluation_examples/examples/tables.json"

SEED = 42
VAL_FRACTION = 0.1
MAX_NEGS_PER_PAIR = 3  # triplet rows emitted per (question, gold table)


def load_schemas():
    path = RAW / "tables.json"
    if not path.exists():
        import urllib.request

        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TABLES_URL, path)
    return {db["db_id"]: db for db in json.loads(path.read_text())}


def describe_tables(db):
    """Return {lowercase original table name: description text} for one database."""
    names = db["table_names_original"]
    cols = defaultdict(list)  # table idx -> [(col idx, natural name)]
    for ci, (ti, name) in enumerate(db["column_names"]):
        if ti >= 0:
            cols[ti].append((ci, name))

    col_table = {ci: ti for ci, (ti, _) in enumerate(db["column_names"])}
    pks = set()
    for pk in db["primary_keys"]:
        pks.update(pk if isinstance(pk, list) else [pk])
    fks = defaultdict(list)  # table idx -> ["col -> other_table.col"]
    for src, dst in db["foreign_keys"]:
        fks[col_table[src]].append(
            f"{db['column_names_original'][src][1]} -> "
            f"{names[col_table[dst]]}.{db['column_names_original'][dst][1]}"
        )

    out = {}
    for ti, orig in enumerate(names):
        natural = db["table_names"][ti]
        header = orig if natural.replace(" ", "_").lower() == orig.lower() else f"{orig} ({natural})"
        col_parts = []
        for ci, name in cols[ti]:
            tags = [db["column_types"][ci]] + (["primary key"] if ci in pks else [])
            col_parts.append(f"{name} ({', '.join(tags)})")
        text = f"Table: {header}\nColumns: {', '.join(col_parts)}"
        if fks[ti]:
            text += f"\nForeign keys: {'; '.join(fks[ti])}"
        out[orig.lower()] = text
    return out


def gold_tables(sql, valid):
    """Table names referenced by the gold SQL (incl. subqueries), restricted to the schema."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        found = {t.name.lower() for t in tree.find_all(exp.Table)}
    except sqlglot.errors.ParseError:
        # Fallback: tokens that follow FROM / JOIN.
        toks = sql.replace("(", " ").replace(")", " ").split()
        found = {toks[i + 1].lower().strip('"`') for i, t in enumerate(toks[:-1]) if t.upper() in ("FROM", "JOIN")}
    return found & valid


def main():
    rng = random.Random(SEED)
    schemas = load_schemas()
    spider = load_dataset("xlangai/spider")

    descs = {db_id: describe_tables(db) for db_id, db in schemas.items()}
    table_id = lambda db_id, t: f"{db_id}.{t}"

    train_dbs = sorted(set(spider["train"]["db_id"]))
    rng.shuffle(train_dbs)
    n_val = round(len(train_dbs) * VAL_FRACTION)
    split_dbs = {"val": set(train_dbs[:n_val]), "train": set(train_dbs[n_val:])}
    split_dbs["test"] = set(spider["validation"]["db_id"])

    # Collect questions per split; merge duplicate questions (same text + db) by unioning gold tables.
    questions = {s: {} for s in split_dbs}
    dropped = Counter()
    for hf_split in ("train", "validation"):
        for row in spider[hf_split]:
            db_id = row["db_id"]
            split = next(s for s, dbs in split_dbs.items() if db_id in dbs)
            gold = gold_tables(row["query"], set(descs[db_id]))
            if not gold:
                dropped[split] += 1
                continue
            key = (db_id, row["question"].strip())
            entry = questions[split].setdefault(key, {"gold": set(), "sql": row["query"]})
            entry["gold"] |= gold

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"seed": SEED, "dropped_no_tables": dict(dropped), "splits": {}}

    with open(OUT / "tables.jsonl", "w") as f:
        for db_id in sorted(descs):
            split = next((s for s, dbs in split_dbs.items() if db_id in dbs), None)
            for t, text in descs[db_id].items():
                f.write(json.dumps({"table_id": table_id(db_id, t), "db_id": db_id, "split": split, "table": t, "text": text}) + "\n")

    for split, qs in questions.items():
        pairs, queries, triplets = [], [], []
        for qi, ((db_id, q), e) in enumerate(sorted(qs.items())):
            qid = f"{split}-{qi}"
            gold = sorted(e["gold"])
            negs = sorted(set(descs[db_id]) - e["gold"])  # other gold tables are never negatives
            queries.append({"qid": qid, "db_id": db_id, "question": q, "sql": e["sql"],
                            "gold_table_ids": [table_id(db_id, t) for t in gold]})
            for t in gold:
                pairs.append({"qid": qid, "db_id": db_id, "anchor": q, "positive": descs[db_id][t],
                              "positive_id": table_id(db_id, t),
                              "hard_negatives": [descs[db_id][n] for n in negs],
                              "hard_negative_ids": [table_id(db_id, n) for n in negs]})
                for n in rng.sample(negs, min(MAX_NEGS_PER_PAIR, len(negs))):
                    triplets.append({"anchor": q, "positive": descs[db_id][t], "negative": descs[db_id][n]})

        write_jsonl(OUT / f"{split}_pairs.jsonl", pairs)
        write_jsonl(OUT / f"{split}_queries.jsonl", queries)
        if split == "train":
            rng.shuffle(triplets)
            write_jsonl(OUT / "train_triplets.jsonl", triplets)

        n_tables = sum(len(descs[d]) for d in split_dbs[split])
        stats["splits"][split] = {
            "databases": len(split_dbs[split]),
            "tables": n_tables,
            "questions": len(queries),
            "pairs": len(pairs),
            "pairs_without_hard_negatives": sum(not p["hard_negatives"] for p in pairs),
            "avg_gold_tables_per_question": round(len(pairs) / len(queries), 2),
            "avg_tables_per_db": round(n_tables / len(split_dbs[split]), 2),
            **({"triplets": len(triplets)} if split == "train" else {}),
            "db_ids": sorted(split_dbs[split]),
        }

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    for s, v in stats["splits"].items():
        print(s, {k: v2 for k, v2 in v.items() if k != "db_ids"})
    print("dropped (no resolvable tables):", dict(dropped))


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
