"""GRPO fine-tuning of Qwen2.5-Coder (4-bit, LoRA) on the NL2SQL prompts with mlx-lm-lora.

Writes models/<run>/data/train.jsonl in mlx-lm-lora's GRPO format, one row per question:
  prompt  the user message from data/rl (retrieved schemas + question + instruction)
  system  Qwen's default system prompt; without it mlx-lm-lora inserts an R1 "<think>/<answer>"
          prompt. With it, prompts are token-identical to eval_sql.py's.
  answer  JSON {"db_id", "gold_sql"}, parsed by the reward (src/grpo_reward.py)
then runs `mlx_lm_lora.train --train-mode grpo` with DEFAULTS. Unknown arguments are passed through
to mlx_lm_lora.train and override DEFAULTS (e.g. --iters 20, --grpo-loss-type dr_grpo).

mlx-lm-lora must be 3.4.x from GitHub (pinned in pyproject.toml): PyPI 3.1.3 computes the GRPO loss
on completion tokens without the prompt, and clips against the reference model with epsilon 1e-4.

Memory: the first full run kernel-panicked the 48 GB Mac (long-prompt batches + MLX pinning up to 37 GB
pushed macOS into swap until its watchdog fired). So the loss runs one completion at a time with
gradient checkpointing, mlx-lm-lora runs under src/mlx_guard.py (soft limit --gpu-memory-gb, process
killed at --gpu-kill-gb), and `caffeinate -i` keeps the Mac awake. Run no other GPU job alongside.

Outputs (models/<run>/): adapters/ (LoRA weights, a checkpoint every --save-every steps; not fused),
rollouts.jsonl (per-step reward stats and sampled completions), train.log, run_config.json.
No validation file is written: mlx-lm-lora's GRPO validation reports only loss. Score checkpoints
with `eval_sql.py --adapter models/<run>/adapters` instead.

Usage:
  uv run python src/train_grpo.py --run-name grpo-smoke --limit 50 --iters 20
  uv run python src/train_grpo.py --run-name grpo-v1 --iters 600
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from eval_sql import DEFAULT_MODEL
from evaluate import ROOT, read_jsonl

QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULTS = {
    "--model": DEFAULT_MODEL,
    "--train-type": "lora",
    "--group-size": "4",
    "--batch-size": "4",              # prompts per step (x group size completions)
    "--max-completion-length": "256",
    "--max-seq-length": "1024",
    "--micro-batch-size": "1",        # completions per loss/grad chunk; same gradient, less memory (4 panicked the Mac)
    "--grad-checkpoint": None,        # flag without a value: recompute activations in backward
    "--temperature": "1.0",           # same as filter_difficulty.py, so its buckets hold
    "--learning-rate": "5e-6",
    "--beta": "0.04",                 # KL weight to the base model
    "--grpo-loss-type": "grpo",
    "--steps-per-report": "10",
    "--save-every": "50",
    "--seed": "0",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--input", default=str(ROOT / "data" / "rl" / "train_filtered.jsonl"))
    p.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    p.add_argument("--sample-every", type=int, default=50, help="log full rollout groups every N steps")
    p.add_argument("--gpu-memory-gb", type=float, default=28, help="MLX soft memory and wired limit")
    p.add_argument("--gpu-kill-gb", type=float, default=34, help="stop training if MLX memory passes this")
    args, passthrough = p.parse_known_args()

    run = ROOT / "models" / args.run_name
    (run / "data").mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input)[:args.limit]
    with open(run / "data" / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["prompt"][0]["content"], "system": QWEN_SYSTEM,
                                "answer": json.dumps({"db_id": r["db_id"], "gold_sql": r["gold_sql"]})}) + "\n")

    # mlx-lm-lora fuses by default, writing a de-quantized 5.8 GB model into the adapter dir; --fuse can
    # only switch that on, so switch it off in a config file (pass --fuse to fuse anyway).
    (run / "config.yaml").write_text("fuse: false\n")
    cmd = [sys.executable, str(ROOT / "src" / "mlx_guard.py"), "-c", str(run / "config.yaml"), "--train", "--train-mode", "grpo",
           "--data", str(run / "data"), "--adapter-path", str(run / "adapters"),
           "--reward-functions-file", str(ROOT / "src" / "grpo_reward.py"), "--reward-functions", "sql_execution"]
    for flag, value in DEFAULTS.items():
        cmd += [flag] if value is None else [flag, value]
    cmd += passthrough  # argparse keeps the last value, so these override DEFAULTS
    (run / "run_config.json").write_text(json.dumps(
        {"input": args.input, "questions": len(rows), "command": cmd[1:],
         "gpu_memory_gb": args.gpu_memory_gb, "gpu_kill_gb": args.gpu_kill_gb}, indent=2))

    rollouts = run / "rollouts.jsonl"
    rollouts.unlink(missing_ok=True)
    env = {**os.environ, "GRPO_LOG": str(rollouts), "GRPO_SAMPLE_EVERY": str(args.sample_every),
           "MLX_SOFT_LIMIT_GB": str(args.gpu_memory_gb), "MLX_HARD_LIMIT_GB": str(args.gpu_kill_gb)}
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", *cmd]  # the panic hit during a sleep/wake under full GPU load
    print(f"{len(rows)} questions -> {run.relative_to(ROOT)}/\n$ {' '.join(cmd)}", flush=True)
    with open(run / "train.log", "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
