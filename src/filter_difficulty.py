"""Keep the RL training questions the base model sometimes gets right.

GRPO scores each answer against the other answers to the same prompt: advantage = (reward - group
mean) / group std. When every answer gets the same reward, every advantage is zero and the prompt
gives no gradient. So for each train question we sample k answers from the base model, score them
with the RL reward (sql_exec.execution_match), and bucket the question:

  easy   all k correct  -> dropped
  hard   none correct   -> dropped, except a random --keep-hard share (seeded)
  mixed  some correct   -> kept

Sample with the temperature RL will use: pass rates depend on it.

Sampling is resumable: every scored question is appended to data/rl/difficulty_samples.jsonl
(completions, rewards, settings, prompt hash). A rerun reuses entries whose settings and prompt
still match and samples the rest.

Outputs (data/rl/):
  difficulty_samples.jsonl  per question: qid, k, temperature, model, prompt_sha, rewards, completions
  train_filtered.jsonl      train.jsonl rows that are kept, plus pass_rate and difficulty
  difficulty_stats.json

Usage:
  uv run python src/filter_difficulty.py --k 8 --temperature 1.0
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

from eval_sql import DEFAULT_MODEL, batch_generate, encode_prompt
from evaluate import ROOT, read_jsonl
from sql_exec import ensure_databases, execution_match, extract_sql

RL = ROOT / "data" / "rl"
SAMPLES = RL / "difficulty_samples.jsonl"


def prompt_sha(row):
    return hashlib.sha1(json.dumps(row["prompt"]).encode()).hexdigest()[:12]


def load_cache(settings):
    """Cached entries made with the same settings, keyed by qid."""
    if not SAMPLES.exists():
        return {}
    cache = {}
    for e in read_jsonl(SAMPLES):
        if all(e[key] == value for key, value in settings.items()):
            cache[e["qid"]] = e
    return cache


def bucket(n_correct, k):
    return "easy" if n_correct == k else "hard" if n_correct == 0 else "mixed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--k", type=int, default=8, help="answers sampled per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="completions generated together")
    p.add_argument("--keep-hard", type=float, default=0.1, help="share of all-wrong questions to keep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="first N train questions only (smoke test)")
    args = p.parse_args()

    ensure_databases()
    rows = read_jsonl(RL / "train.jsonl")[:args.limit]
    settings = {"model": args.model, "k": args.k, "temperature": args.temperature, "max_tokens": args.max_tokens}
    cache = load_cache(settings)
    todo = [r for r in rows if r["qid"] not in cache or cache[r["qid"]]["prompt_sha"] != prompt_sha(r)]
    print(f"{len(rows)} train questions: {len(rows) - len(todo)} cached, {len(todo)} to sample (k={args.k}, "
          f"temperature={args.temperature})")

    if todo:
        mx.random.seed(args.seed)
        model, tokenizer = load(args.model)
        sampler = make_sampler(temp=args.temperature)
        per_chunk = max(1, args.batch_size // args.k)  # questions per batch_generate call
        start = time.time()
        with open(SAMPLES, "a") as f:
            for c in range(0, len(todo), per_chunk):
                chunk = todo[c:c + per_chunk]
                prompts = [encode_prompt(tokenizer, r["prompt"]) for r in chunk for _ in range(args.k)]
                texts = batch_generate(model, tokenizer, prompts, max_tokens=args.max_tokens, sampler=sampler,
                                       completion_batch_size=args.batch_size).texts
                for i, r in enumerate(chunk):
                    completions = texts[i * args.k:(i + 1) * args.k]
                    rewards = [int(execution_match(r["db_id"], extract_sql(t), r["gold_sql"])) for t in completions]
                    entry = {"qid": r["qid"], **settings, "prompt_sha": prompt_sha(r),
                             "rewards": rewards, "completions": completions}
                    cache[r["qid"]] = entry
                    f.write(json.dumps(entry) + "\n")
                f.flush()
                mx.clear_cache()  # freed buffers of every padded batch shape stay cached otherwise -> Metal OOM after ~1 h
                done = c + len(chunk)
                eta = (time.time() - start) / done * (len(todo) - done)
                print(f"  {done}/{len(todo)} questions sampled, ~{eta / 60:.0f} min left "
                      f"(GPU memory: {mx.get_active_memory() / 2**30:.1f} GB active, "
                      f"{mx.get_cache_memory() / 2**30:.1f} GB cached)", flush=True)

    rng = random.Random(args.seed)
    kept, buckets, hist = [], Counter(), Counter()
    for r in rows:
        n = sum(cache[r["qid"]]["rewards"])
        b = bucket(n, args.k)
        buckets[b] += 1
        hist[n] += 1
        if b == "mixed" or (b == "hard" and rng.random() < args.keep_hard):
            kept.append({**r, "pass_rate": n / args.k, "difficulty": b})

    suffix = f"_limit{args.limit}" if args.limit else ""
    with open(RL / f"train_filtered{suffix}.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    kept_by = Counter(r["difficulty"] for r in kept)
    stats = {**settings, "keep_hard": args.keep_hard, "seed": args.seed, "questions": len(rows),
             "mean_pass_rate": round(sum(n * c for n, c in hist.items()) / (args.k * len(rows)), 4),
             "buckets": dict(buckets), "kept": len(kept), "kept_by_bucket": dict(kept_by),
             "n_correct_histogram": {n: hist[n] for n in range(args.k + 1)}}
    (RL / f"difficulty_stats{suffix}.json").write_text(json.dumps(stats, indent=2))

    print(f"\nmean pass rate {stats['mean_pass_rate']:.1%} | "
          + " | ".join(f"{b} {buckets[b]} ({buckets[b] / len(rows):.0%})" for b in ("easy", "mixed", "hard")))
    print("correct answers out of", args.k, "->", {n: hist[n] for n in range(args.k + 1)})
    print(f"kept {len(kept)}: {kept_by['mixed']} mixed + {kept_by['hard']} hard -> "
          f"{(RL / f'train_filtered{suffix}.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
