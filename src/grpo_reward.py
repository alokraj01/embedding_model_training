"""GRPO reward functions for mlx-lm-lora (`--reward-functions-file src/grpo_reward.py`).

Each training row's answer field is a JSON string {"db_id": ..., "gold_sql": ...}.

  sql_execution  1.0 if the completion's ```sql block returns the gold rows (sql_exec.execution_match),
                 else 0.0: no block, SQL error, timeout (2 s) or a different result.
  sql_format     0.1 for a reply that is exactly one ```sql block and nothing else. Optional: add it
                 with --reward-functions sql_execution,sql_format and keep it small, or the model
                 optimizes format instead of correctness.

sql_execution also writes one JSON line per training step to $GRPO_LOG (train_grpo.py points it at
models/<run>/rollouts.jsonl): mean reward, share of groups whose rewards are all equal (those give
no gradient), mean completion length, peak MLX memory since the previous step (the previous step's
loss/backward plus this step's rollouts), and every $GRPO_SAMPLE_EVERY steps the full group of the
first few prompts, to read for reward hacking.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # loaded by file path, so make src/ importable

import mlx.core as mx
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
from tqdm import tqdm

from sql_exec import execution_match, extract_sql

CLEAN = re.compile(r"```sql\n[^`]+\n```", re.I)
SAMPLE_PROMPTS = 2  # prompts whose whole group is logged on sample steps
_step = 0


def _groups(prompts, answers):
    """Index runs of identical (prompt, answer): the trainer passes completions grouped by prompt."""
    groups, prev = [], None
    for i, key in enumerate(zip(prompts, answers)):
        if key != prev:
            groups.append([])
            prev = key
        groups[-1].append(i)
    return groups


def _log(prompts, completions, answers, rewards):
    global _step
    _step += 1
    groups = _groups(prompts, answers)
    entry = {
        "step": _step,
        "reward_mean": round(sum(rewards) / len(rewards), 4),
        "all_same_groups": round(sum(len({rewards[i] for i in g}) == 1 for g in groups) / len(groups), 4),
        "completion_chars": round(sum(map(len, completions)) / len(completions), 1),
        "no_sql_block": round(sum(extract_sql(c) is None for c in completions) / len(completions), 4),
        "gpu_peak_gb": round(mx.get_peak_memory() / 2**30, 2),
    }
    mx.reset_peak_memory()
    every = int(os.environ.get("GRPO_SAMPLE_EVERY", 50))
    if _step == 1 or _step % every == 0:
        entry["samples"] = [{"question": prompts[g[0]].rsplit("Question: ", 1)[-1].split("\n")[0],
                             "gold_sql": json.loads(answers[g[0]])["gold_sql"],
                             "completions": [{"reward": rewards[i], "text": completions[i]} for i in g]}
                            for g in groups[:SAMPLE_PROMPTS]]
    with open(os.environ.get("GRPO_LOG", "grpo_log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    if _step % 10 == 0:
        tqdm.write(f"[rollouts] step {_step}: reward {entry['reward_mean']:.3f}, "
                   f"all-same groups {entry['all_same_groups']:.0%}, {entry['completion_chars']:.0f} chars, "
                   f"peak {entry['gpu_peak_gb']:.1f} GB")


@register_reward_function()
def sql_execution(prompts, completions, answer, types=None):
    gold = [json.loads(a) for a in answer]
    rewards = [float(execution_match(g["db_id"], extract_sql(c), g["gold_sql"])) for c, g in zip(completions, gold)]
    _log(prompts, completions, answer, rewards)
    return rewards


@register_reward_function()
def sql_format(prompts, completions, answer, types=None):
    return [0.1 if CLEAN.fullmatch(c.strip()) else 0.0 for c in completions]
