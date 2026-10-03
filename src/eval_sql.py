"""Execution accuracy of an LLM on the RL prompts (data/rl/{split}.jsonl), greedy decoding with MLX.

A prediction is correct when the SQL in its last ```sql block returns the gold rows
(sql_exec.execution_match, the same check as the RL reward). Questions whose gold query is not
usable (gold_status != "ok") are left out of every metric.

  ex_end_to_end       share correct; questions where retrieval missed a gold table count as wrong
  ex_given_retrieval  share correct among questions where retrieval found every gold table
                      (the setting RL trains on)
  no_sql_block        replies without a ```sql block
  sql_error           SQL that fails to run (syntax, unknown table/column, timeout)
  truncated           replies that hit --max-tokens

Usage:
  uv run python src/eval_sql.py --split test
  uv run python src/eval_sql.py --split test --adapter models/<run>/adapters --tag <run>
"""

import argparse
import importlib
import json
import re
import time
from collections import defaultdict

import mlx.core as mx
from mlx_lm import load

from evaluate import RESULTS, ROOT, read_jsonl
from sql_exec import ensure_databases, execute, execution_match, extract_sql

batch_generate = importlib.import_module("mlx_lm.generate").batch_generate  # mlx_lm.generate is shadowed by a function

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"


def encode_prompt(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def generate(model, tokenizer, rows, max_tokens, batch_size):
    """Greedy completions for each row's chat prompt, in row order, with generated-token counts."""
    prompts = [encode_prompt(tokenizer, r["prompt"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # similar lengths per batch -> less padding
    texts, n_tokens = [None] * len(rows), [0] * len(rows)
    start = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        out = batch_generate(model, tokenizer, [prompts[i] for i in idx], max_tokens=max_tokens,
                             completion_batch_size=batch_size, return_token_ids=True)
        for i, text, toks in zip(idx, out.texts, out.token_ids):
            texts[i], n_tokens[i] = text, len(toks)
        mx.clear_cache()  # see filter_difficulty.py
        done = b + len(idx)
        print(f"  {done}/{len(rows)} generated ({(time.time() - start) / done:.2f}s per question)", flush=True)
    return texts, n_tokens


def rate(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


def summarize(per_query):
    usable = [r for r in per_query if r["gold_status"] == "ok"]
    complete = [r for r in usable if r["retrieval_complete"]]
    by_gold = defaultdict(list)
    for r in complete:
        by_gold[str(min(r["n_gold_tables"], 3)).replace("3", "3+")].append(r)
    return {
        "questions": len(per_query),
        "usable_gold": len(usable),
        "retrieval_complete": len(complete),
        "ex_end_to_end": rate(usable, "correct"),
        "ex_given_retrieval": rate(complete, "correct"),
        "no_sql_block": rate(usable, "no_sql_block"),
        "sql_error": rate(usable, "sql_error"),
        "truncated": rate(usable, "truncated"),
        "ex_given_retrieval_by_gold_count": {g: {"n": len(rs), "ex": rate(rs, "correct")}
                                             for g, rs in sorted(by_gold.items())},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="LoRA adapter directory (e.g. after RL)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    p.add_argument("--tag", default=None, help="name for the results file (default: derived from model)")
    args = p.parse_args()

    ensure_databases()
    rows = read_jsonl(ROOT / "data" / "rl" / f"{args.split}.jsonl")[:args.limit]
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    print(f"{args.model}{f' + {args.adapter}' if args.adapter else ''}: {len(rows)} {args.split} questions")
    texts, n_tokens = generate(model, tokenizer, rows, args.max_tokens, args.batch_size)

    per_query = []
    for r, text, n in zip(rows, texts, n_tokens):
        sql = extract_sql(text)
        err = execute(r["db_id"], sql)[1] if sql else None
        per_query.append({
            "qid": r["qid"], "db_id": r["db_id"], "question": r["question"], "gold_sql": r["gold_sql"],
            "gold_status": r["gold_status"], "retrieval_complete": r["retrieval_complete"],
            "n_gold_tables": len(r["gold_tables"]), "completion": text, "pred_sql": sql,
            "correct": execution_match(r["db_id"], sql, r["gold_sql"]),
            "no_sql_block": sql is None, "sql_error": err is not None, "error": err,
            "truncated": n >= args.max_tokens,
        })

    summary = summarize(per_query)
    res = {"model": args.model, "adapter": args.adapter, "split": args.split, "decoding": "greedy",
           "max_tokens": args.max_tokens, "summary": summary, "per_query": per_query}
    tag = args.tag or re.sub(r"[^\w.-]+", "_", args.model.strip("/").split("/")[-1])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sql__{tag}__{args.split}{f'__limit{args.limit}' if args.limit else ''}.json"
    out.write_text(json.dumps(res, indent=1))

    s = summary
    print(f"\nexecution accuracy ({s['usable_gold']} questions with a usable gold query)")
    print(f"  end to end (retrieval misses count as wrong): {s['ex_end_to_end']:.1%}")
    print(f"  given retrieval found all gold tables ({s['retrieval_complete']}): {s['ex_given_retrieval']:.1%}")
    for g, m in s["ex_given_retrieval_by_gold_count"].items():
        print(f"    {g} gold tables ({m['n']}): {m['ex']:.1%}")
    print(f"  no sql block {s['no_sql_block']:.1%} | sql error {s['sql_error']:.1%} | truncated {s['truncated']:.1%}")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
