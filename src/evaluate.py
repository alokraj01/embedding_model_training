"""Table-retrieval evaluation: rank candidate tables for each question, report recall@k.

Each question is ranked against a pool of tables:
  --pool split : all tables from the split's databases (default; test = 81 tables)
  --pool all   : every table in tables.jsonl (876), a harder setting

Metrics, averaged over questions:
  recall@k   share of the question's gold tables found in the top k
  complete@k share of questions with *all* gold tables in the top k (what SQL generation needs)
  mrr        reciprocal rank of the first gold table

Each metric is reported at the full embedding size and at truncated (Matryoshka) sizes.

Usage:
  uv run python src/evaluate.py --model BAAI/bge-small-en-v1.5 --split test
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# Instruction bge v1.5 recommends for short-query -> passage retrieval.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
KS = (1, 3, 5, 10)
DIMS = (768, 512, 384, 256, 128, 64)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_split(split, pool):
    queries = read_jsonl(DATA / f"{split}_queries.jsonl")
    tables = read_jsonl(DATA / "tables.jsonl")
    if pool == "split":
        tables = [t for t in tables if t["split"] == split]
    return queries, tables


def rank_metrics(ranks):
    """Metrics from the 0-based rank of each gold table, one list per question."""
    m = {}
    for k in KS:
        m[f"recall@{k}"] = float(np.mean([np.mean([r < k for r in rs]) for rs in ranks]))
    for k in (3, 5):
        m[f"complete@{k}"] = float(np.mean([max(rs) < k for rs in ranks]))
    m["mrr"] = float(np.mean([1 / (min(rs) + 1) for rs in ranks]))
    return m


def score(q_emb, t_emb, queries, table_ids, dims):
    """Return ({dim: metrics}, {gold-table count: metrics} at full size, top-10 table ids per query at full size)."""
    gold_idx = [[table_ids.index(g) for g in q["gold_table_ids"]] for q in queries]
    results, by_gold, top_full = {}, None, None
    for dim in dims:
        q = q_emb[:, :dim] / np.linalg.norm(q_emb[:, :dim], axis=1, keepdims=True)
        t = t_emb[:, :dim] / np.linalg.norm(t_emb[:, :dim], axis=1, keepdims=True)
        order = np.argsort(-(q @ t.T), axis=1)
        # rank (0-based) of each gold table for each question
        ranks = [[int(np.where(order[i] == g)[0][0]) for g in gold] for i, gold in enumerate(gold_idx)]
        results[dim] = rank_metrics(ranks)
        if top_full is None:
            top_full = [[table_ids[j] for j in row[:10]] for row in order]
            groups = {"1": [], "2": [], "3+": []}
            for rs in ranks:
                groups[str(len(rs)) if len(rs) < 3 else "3+"].append(rs)
            by_gold = {g: {"n": len(rs), **rank_metrics(rs)} for g, rs in groups.items() if rs}
    return results, by_gold, top_full


def evaluate(model_name, split, pool, query_prompt, batch_size=64):
    queries, tables = load_split(split, pool)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    q_emb = model.encode([query_prompt + q["question"] for q in queries], batch_size=batch_size,
                         normalize_embeddings=True, show_progress_bar=True)
    t_emb = model.encode([t["text"] for t in tables], batch_size=batch_size,
                         normalize_embeddings=True, show_progress_bar=True)

    full = q_emb.shape[1]
    dims = [full] + [d for d in DIMS if d < full]
    metrics, by_gold, top10 = score(q_emb, t_emb, queries, [t["table_id"] for t in tables], dims)

    return {
        "model": model_name, "split": split, "pool": pool, "query_prompt": query_prompt,
        "n_queries": len(queries), "n_tables": len(tables), "embedding_dim": full,
        "metrics": {str(d): m for d, m in metrics.items()},
        "metrics_by_gold_count": by_gold,
        "per_query": [{"qid": q["qid"], "question": q["question"], "gold": q["gold_table_ids"], "top10": top}
                      for q, top in zip(queries, top10)],
    }


def print_table(res):
    print(f"\n{res['model']} | split={res['split']} pool={res['pool']} "
          f"({res['n_queries']} questions, {res['n_tables']} tables)")
    cols = [f"recall@{k}" for k in KS] + ["complete@3", "complete@5", "mrr"]
    print(f"{'dim':>5} " + " ".join(f"{c:>10}" for c in cols))
    for dim, m in res["metrics"].items():
        print(f"{dim:>5} " + " ".join(f"{m[c]:>10.4f}" for c in cols))

    print(f"\nby number of gold tables ({res['embedding_dim']} dims)")
    print(f"{'gold':>5} {'questions':>10} " + " ".join(f"{c:>10}" for c in cols))
    for g, m in res["metrics_by_gold_count"].items():
        print(f"{g:>5} {m['n']:>10} " + " ".join(f"{m[c]:>10.4f}" for c in cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--pool", default="split", choices=["split", "all"])
    p.add_argument("--no-prompt", action="store_true", help="embed questions without the bge query instruction")
    p.add_argument("--tag", default=None, help="name for the results file (default: derived from model)")
    args = p.parse_args()

    prompt = "" if args.no_prompt else BGE_QUERY_PROMPT
    res = evaluate(args.model, args.split, args.pool, prompt)
    print_table(res)

    tag = args.tag or re.sub(r"[^\w.-]+", "_", args.model.strip("/").split("/")[-1])
    if args.no_prompt:
        tag += "_noprompt"
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{tag}__{args.split}__{args.pool}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
