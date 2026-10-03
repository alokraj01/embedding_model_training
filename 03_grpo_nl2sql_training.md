# GRPO training for NL2SQL

The second stage of this project trains a small LLM to write SQL for a question, using the tables the retriever from stage one selects. Training uses reinforcement learning with **GRPO** (Group Relative Policy Optimization): the model writes several SQL answers per question, each answer is run against the real database, and the model is pushed toward the answers that returned the right rows.

This document covers every step from a trained retriever to a GRPO-trained model, then explains the reward, the loss, how the weights are updated, and how to read the training output. It describes the code in `src/` with `mlx-lm` 0.32 and `mlx-lm-lora` 3.4.3 on an Apple M5 Pro (48 GB). Stage one (the retriever) is covered in [02_training_and_loss.md](02_training_and_loss.md).

- [1. Pipeline overview](#1-pipeline-overview)
- [2. Step by step](#2-step-by-step)
- [3. The reward](#3-the-reward)
- [4. One GRPO step, end to end](#4-one-grpo-step-end-to-end)
- [5. Advantages](#5-advantages)
- [6. The loss function](#6-the-loss-function)
- [7. How the parameters are updated](#7-how-the-parameters-are-updated)
- [8. Training parameters](#8-training-parameters)
- [9. Memory on Apple Silicon](#9-memory-on-apple-silicon)
- [10. Reading the training output](#10-reading-the-training-output)
- [11. Results so far](#11-results-so-far)
- [12. mlx-lm-lora pitfalls](#12-mlx-lm-lora-pitfalls)
- [13. Known limitations](#13-known-limitations)

---

## 1. Pipeline overview

```
trained retriever (stage 1)
   │
   ▼  build_rl_prompts.py      retrieve top-5 tables per question, render schemas, check every gold query
data/rl/{train,val,test}.jsonl
   │
   ├─▶ eval_sql.py             baseline: greedy accuracy of the untrained LLM
   │
   ▼  filter_difficulty.py     sample 8 answers per train question, keep the ones the model sometimes solves
data/rl/train_filtered.jsonl
   │
   ▼  train_grpo.py            GRPO with LoRA on the 4-bit model (mlx-lm-lora, under mlx_guard.py)
models/<run>/adapters/
   │
   ▼  eval_sql.py --adapter    greedy accuracy of the trained model, same prompts and scoring as the baseline
```

| File | Role |
|---|---|
| [src/sql_exec.py](src/sql_exec.py) | Downloads the Spider SQLite databases; runs SQL read-only with a time limit; execution match; gold-query checks |
| [src/build_rl_prompts.py](src/build_rl_prompts.py) | Builds the prompts from retrieved tables |
| [src/eval_sql.py](src/eval_sql.py) | Greedy generation with MLX and execution accuracy |
| [src/filter_difficulty.py](src/filter_difficulty.py) | Difficulty filter for the training set |
| [src/train_grpo.py](src/train_grpo.py) | Writes the training data and launches mlx-lm-lora |
| [src/grpo_reward.py](src/grpo_reward.py) | The reward function mlx-lm-lora calls; also logs rollouts |
| [src/mlx_guard.py](src/mlx_guard.py) | Caps MLX memory so a large step can't freeze macOS |

The policy model is **Qwen2.5-Coder-3B-Instruct in 4-bit** (`mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`): 36 layers, hidden size 2048, vocabulary 151,936, 4-bit weights with group size 64.

## 2. Step by step

Run everything from the project root, and run **one GPU job at a time** (see [section 9](#9-memory-on-apple-silicon)).

### Step 1. Download the Spider databases

```bash
uv run python src/sql_exec.py
```

Downloads the 166 Spider SQLite databases (about 470 MB) from the Hugging Face mirror `minktn/spider-data` into `data/raw/spider_db/spider_data/database/<db_id>/<db_id>.sqlite`. Every table and column in `data/raw/tables.json` was checked against the files: no mismatches. Step 2 runs this automatically if the databases are missing.

### Step 2. Build the prompts

```bash
uv run python src/build_rl_prompts.py      # --model models/bge-small-mnrl-matryoshka/final --k 5 --pool all
```

For every question, the retriever ranks all 876 tables and the top 5 go into the prompt as one-line `CREATE TABLE` statements with the **original** column names. The embedding text in `tables.jsonl` uses Spider's natural-language names ("singer id"), which would produce SQL that can't run.

```
-- Database: battle_death
CREATE TABLE ship (lost_in_battle number, id number PRIMARY KEY, name text, tonnage text, ..., FOREIGN KEY (lost_in_battle) REFERENCES battle(id));
CREATE TABLE battle (id number PRIMARY KEY, name text, date text, bulgarian_commander text, latin_commander text, result text);

-- Database: ship_mission
CREATE TABLE ship (Ship_ID number PRIMARY KEY, Name text, Type text, Nationality text, Tonnage number);

Question: How many battles did not lose any ship with tonnage '225'?
Write one SQLite query that answers the question. Use tables from one database only. Reply with only the SQL in a ```sql code block.
```

Retrieval runs over every table, so the top 5 nearly always spans several databases; tables are grouped under `-- Database:` headers. Identifiers are quoted only when SQLite rejects them bare (e.g. a column named `From`). Prompts are short: about 255 tokens at the median, 511 at most in the training set, including the chat template.

The script also runs every gold query and labels it:

| `gold_status` | Meaning | Why it matters |
|---|---|---|
| `ok` | Returns at least one row with a non-NULL value | Usable reward target |
| `empty` | Returns no rows | Any query that returns nothing would score 1 |
| `null_only` | Every value is NULL | Same problem |
| `zero` | A single `0` | Any over-filtered `count(*)` would score 1 |
| `error` | Fails to run | Broken Spider annotation; can never be matched |

**Train keeps only rows where retrieval found every needed table and the gold is `ok`.** If retrieval missed a table, the model can never be right and the example only wastes rollouts. Val and test keep every row, with `retrieval_complete` and `gold_status` fields, so evaluation counts retrieval misses as failures.

| Split | Questions | Retrieval missed a needed table | Ceiling | Rows written |
|---|---|---|---|---|
| train | 6,366 | 1,040 | 83.7% | 5,084 (after also dropping non-`ok` golds) |
| val | 625 | 107 | 82.9% | 625 |
| test | 1,034 | 205 | 80.2% | 1,034 |

The **ceiling** is the share of questions where retrieval found every needed table: the best end-to-end accuracy any SQL model could reach with this retriever.

Outputs: `data/rl/{train,val,test}.jsonl`, `data/rl/stats.json`.

### Step 3. Baseline

```bash
uv run python src/eval_sql.py --split test
uv run python src/eval_sql.py --split val
```

Greedy generation with the untrained model on every prompt, scored with the same execution match as the reward. Rows whose gold isn't `ok` are excluded.

| Split | Correct, given retrieval found all needed tables | Correct end to end | SQL that fails to run |
|---|---|---|---|
| test | **66.0%** (773 questions) | 54.5% | 29.5% |
| val | **59.5%** (499 questions) | 50.2% | 31.4% |

Batched greedy decoding gives the same outputs as batch size 1, so these numbers are reproducible. The most common failure is mixing databases: 71 of the 773 test questions fail because the model uses a table from another database in the prompt, or a database name as a table.

Outputs: `results/sql__Qwen2.5-Coder-3B-Instruct-4bit__{test,val}.json`.

### Step 4. Difficulty filter

```bash
uv run python src/filter_difficulty.py      # --k 8 --temperature 1.0
```

Samples 8 answers per training question from the base model, at the temperature training will use, and scores them:

| Bucket | Correct answers out of 8 | Questions | Kept |
|---|---|---|---|
| easy | 8 | 901 | none |
| mixed | 1–7 | 3,071 | all |
| hard | 0 | 1,112 | 125 (random 10%) |

Why this matters is explained in [section 5](#5-advantages): a question the model always or never gets right gives no learning signal. The hard bucket also contains broken Spider labels (e.g. gold SQL joining on `T2.actid = T2.actid`), which is one more reason to keep only a few.

The base model's mean pass rate on the 3,196 kept questions is **0.514**; that's the expected training reward before any learning. About an hour on the M5 Pro. The script is resumable: samples are cached and reused while the model, k, temperature and prompts are unchanged.

Outputs: `data/rl/train_filtered.jsonl` (the training set), `data/rl/difficulty_samples.jsonl` (every sampled answer), `data/rl/difficulty_stats.json`.

### Step 5. Smoke test

```bash
uv run python src/train_grpo.py --run-name grpo-smoke --limit 50 --iters 20
```

20 steps on 50 questions, about 5 minutes. Check that it doesn't crash, that the reward isn't stuck at 0, and that memory is fine.

### Step 6. Train

```bash
uv run python src/train_grpo.py --run-name grpo-v1 --iters 600
uv run python src/train_grpo.py --run-name grpo-drgrpo --iters 600 --grpo-loss-type dr_grpo   # second run
```

Any argument `train_grpo.py` doesn't know is passed to mlx-lm-lora and overrides the defaults in [section 8](#8-training-parameters). 600 steps take about 2 hours (about 12 s per step).

Outputs in `models/<run>/`:

| File | Contents |
|---|---|
| `data/train.jsonl` | The training data in mlx-lm-lora's format |
| `config.yaml` | `fuse: false` (see [section 12](#12-mlx-lm-lora-pitfalls)) |
| `run_config.json` | The exact command and memory limits |
| `adapters/adapters.safetensors` | Latest LoRA weights, plus `adapter_config.json` |
| `adapters/0000050_adapters.safetensors`, … | A checkpoint every 50 steps |
| `rollouts.jsonl` | Per-step reward statistics, peak memory, and sampled answers every 50 steps |
| `train.log` | Full trainer output |

### Step 7. Evaluate

```bash
uv run python src/eval_sql.py --split val  --adapter models/grpo-v1/adapters --tag grpo-v1
uv run python src/eval_sql.py --split test --adapter models/grpo-v1/adapters --tag grpo-v1
```

Same prompts, greedy decoding and scoring as the baseline, so the numbers compare directly. Choose the checkpoint on val and report test once. `--adapter` loads `adapters.safetensors` (the latest weights); to score an earlier checkpoint, copy it as `adapters.safetensors` into its own folder with `adapter_config.json`.

## 3. The reward

[src/grpo_reward.py](src/grpo_reward.py) `sql_execution`, using [src/sql_exec.py](src/sql_exec.py):

1. Take the SQL from the last ```` ```sql ```` block. No block: reward **0**.
2. Run it on the question's database, opened **read-only**, with a **2 s** limit (the slowest gold query takes 0.4 s) and a 10,000-row cap. Error or timeout: **0**.
3. Run the gold SQL (cached after the first time) and compare the rows. Same rows: **1**, otherwise **0**.

How rows are compared:

- **Order** counts only when the gold query has a top-level `ORDER BY`; otherwise rows are compared as multisets.
- **Column order** doesn't count: `SELECT name, country` matches `SELECT country, name`.
- **Floats** are rounded to 6 places.

Each row's `answer` field carries what the reward needs, as a JSON string: `{"db_id": "battle_death", "gold_sql": "SELECT ..."}`.

**Guards against reward hacking:**

| Hole | Guard | Check |
|---|---|---|
| Gold returns nothing, so any broken query matches | `empty`, `null_only` and `zero` golds removed from training | `SELECT 1 WHERE 0`, `SELECT NULL` and `SELECT 0` match 0 of 5,084 train rows |
| Model writes `DROP TABLE` or `INSERT` | Databases opened read-only | Both score 0; the table is intact afterwards |
| Runaway query (e.g. infinite recursive CTE) | 2 s time limit | Stopped and scored 0 |
| Reward for format instead of correctness | No format bonus | `sql_format` (0.1 for a clean reply) exists but is off: the base model already formats every reply correctly |

Execution match isn't perfect. `SELECT 1` or `SELECT 2` still matches about 1.4% of train rows (where the true answer is that number), and another question's gold SQL from the same database matches 0.5%: Spider's databases are small enough that different queries sometimes return the same rows. Both are too rare to exploit.

## 4. One GRPO step, end to end

With the defaults, one step handles 4 prompts:

```
4 prompts (schemas + question)
   │
   ▼  sample 4 answers each, temperature 1.0, max 256 tokens
16 completions                                   e.g. ```sql\nSELECT count(*) FROM singer\n```
   │
   ▼  run each on its database, compare with the gold rows
16 rewards, in 4 groups                          [1,0,0,1]  [1,1,1,1]  [0,1,0,0]  [1,1,0,1]
   │
   ▼  normalize within each group
16 advantages                                    [+1,−1,−1,+1]  [0,0,0,0]  [−.58,+1.73,−.58,−.58]  ...
   │
   ▼  log-probability of every completion token given its prompt, policy and reference model
loss = −advantage × log p(token)  +  β × KL     averaged as in section 6
   │
   ▼  backward (one completion at a time), Adam update of the LoRA weights
next step
```

**Sampling.** Each prompt is processed once, and its cache is shared by the 4 samples. Generation stops at the end-of-sequence token or after 256 tokens. The sampled token ids, including the end-of-sequence token, are what the loss is computed on, so decoding and re-encoding text can't introduce mismatches.

**Temperature 1.0** gives the variety GRPO needs: it learns only from *differences* within a group. It also matches the difficulty filter, so the filter's "mixed" label holds during training.

**There is no value network.** PPO trains a second network to predict expected reward; GRPO uses the group's mean reward instead. That's what makes it cheap enough for a laptop.

**Each batch of samples is used for exactly one update** (on-policy), then discarded.

## 5. Advantages

Within each group of 4 answers to the same question:

$$A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + 10^{-4}}$$

(`std` is the population standard deviation.)

| Group rewards | Mean | Std | Advantages | Meaning |
|---|---|---|---|---|
| `[1, 0, 0, 1]` | 0.5 | 0.5 | `[+1, −1, −1, +1]` | Reinforce the 2 right answers, discourage the 2 wrong ones |
| `[1, 0, 0, 0]` | 0.25 | 0.433 | `[+1.73, −0.58, −0.58, −0.58]` | A rare success gets a large push up |
| `[0, 1, 1, 1]` | 0.75 | 0.433 | `[−1.73, +0.58, +0.58, +0.58]` | A rare failure gets a large push down |
| `[1, 1, 1, 1]` | 1 | 0 | `[0, 0, 0, 0]` | **No signal** |
| `[0, 0, 0, 0]` | 0 | 0 | `[0, 0, 0, 0]` | **No signal** |

Two consequences:

- **Only mixed groups teach anything.** That's why the difficulty filter drops questions the model always or never solves. Even on mixed questions, a group of 4 comes out all-same often: with the filter's pass rates, about 36% of the time at group size 4, 23% at 6, and 16% at 8. The share grows as the model improves.
- **Advantages within a group sum to zero.** The model isn't rewarded for being right in absolute terms, only for being better than its own other attempts at the same question. That makes the signal comparable across easy and hard questions.

## 6. The loss function

Implemented in mlx-lm-lora's `grpo_loss` / `_grpo_objective`.

### Log-probabilities

Each completion is appended to its prompt, the model runs over the full sequence, and the log-probability of each **completion** token given everything before it is taken:

$$\ell_{i,t} = \log \pi_\theta(o_{i,t} \mid \text{prompt}_i, o_{i,<t})$$

The same is computed with the frozen reference model (the base model without LoRA), giving $\ell^{\text{ref}}_{i,t}$.

### Per-token loss

$$L_{i,t} = -\min\big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon_{\text{high}})\,A_i\big) + \beta\,\mathrm{KL}_{i,t}$$

**Ratio.** $\rho_{i,t} = \exp(\ell_{i,t} - \operatorname{stopgrad}(\ell_{i,t}))$, which is exactly 1 in value but carries the gradient of $\ell_{i,t}$. Because samples are used once (on-policy), the policy that generated them is the current one. So:

- the `min`/`clip` never binds (the clip ratios in the reports stay at 0, and `--epsilon` has no effect);
- the first term reduces to $-A_i\,\ell_{i,t}$ for gradient purposes.

**Policy gradient.** $\nabla L_{i,t} = -A_i \nabla \ell_{i,t}$. Minimizing the loss raises the probability of every token in an above-average answer and lowers it in a below-average one. A correct `GROUP BY` clause is reinforced; an invented column name is discouraged.

**KL penalty.** Per token, with $\delta = \ell^{\text{ref}}_{i,t} - \ell_{i,t}$ (capped at 20):

$$\mathrm{KL}_{i,t} = e^{\delta} - \delta - 1$$

This is the low-variance "k3" estimator of $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. It's 0 when the two models agree and grows as they drift apart. Its gradient with respect to $\ell_{i,t}$ is $1 - \pi_{\text{ref}}/\pi_\theta$, which pulls each token's probability back toward the base model. $\beta = 0.04$ keeps the model from forgetting how to write SQL in general while it chases reward.

### Averaging over tokens and answers

`--grpo-loss-type` decides how the per-token losses become one number, for $N$ completions with $|o_i|$ tokens each:

| Type | Formula | Effect |
|---|---|---|
| `grpo` (default) | $\frac{1}{N}\sum_i \frac{1}{\lvert o_i\rvert}\sum_t L_{i,t}$ | Every answer weighs the same, so each token of a long answer weighs less. A long *wrong* answer is penalized less per token than a short one: a known bias toward longer outputs |
| `bnpo` | $\frac{1}{\sum_i \lvert o_i\rvert}\sum_{i,t} L_{i,t}$ | Every token in the batch weighs the same |
| `dr_grpo` | $\frac{1}{N \cdot 256}\sum_{i,t} L_{i,t}$ | Constant denominator (256 = max completion length), removing the length bias |

In this implementation `dr_grpo` changes only this averaging; advantages are still divided by the group std.

### Why the printed loss is about 0.001

Because $\rho = 1$ and each group's advantages sum to zero, the policy term's *value* averages to roughly zero, even though its *gradient* is not zero. What remains is $\beta \times$ KL: for example, $0.04 \times 0.03 \approx 0.0012$. So the loss value says almost nothing about progress. Watch the reward.

## 7. How the parameters are updated

### What is trained: LoRA on a frozen 4-bit model

The base weights stay frozen in 4-bit. Every linear layer in all 36 transformer blocks gets a low-rank adapter:

$$y = W_{\text{4-bit}}\,x + s \cdot B(Ax)$$

- $A$ is rank × input and $B$ is output × rank, with rank $r = 8$ and scale $s = 10$.
- $B$ starts at zero, so training starts exactly at the base model.
- The adapted layers are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

Per block that's $8 \times [(2048+2048)\cdot 2 + (2048+256)\cdot 2 + (2048+11008)\cdot 3] = 415{,}744$ parameters, **14,966,784 in total, 0.485% of the model**.

### One update

1. **Micro-batches.** The 16 completions go through forward and backward **one at a time** (`--micro-batch-size 1`). Each one's gradient is weighted by its share of the objective (1/16 for `grpo`) and summed. The result equals the gradient of the full batch: micro-batching changes memory use, not the training.
2. **Gradient checkpointing.** Layer activations aren't kept for the backward pass; they're recomputed. Slower, much less memory.
3. **Adam** with a constant learning rate of 5e-6 updates only $A$ and $B$. There's one optimizer update per step (no gradient accumulation across steps).
4. **The reference model** isn't updated. It's a separate frozen copy of the base model, used only for the KL term.

### Checkpoints

Every 50 steps the LoRA weights are saved as `adapters/00000NN_adapters.safetensors`, and `adapters.safetensors` is overwritten with the latest. Each file is about 60 MB. The merged full model isn't written (see [section 12](#12-mlx-lm-lora-pitfalls)); step 9 of the project plan fuses it deliberately when shipping.

## 8. Training parameters

Defaults in `DEFAULTS` in [src/train_grpo.py](src/train_grpo.py):

| Parameter | Value | Why |
|---|---|---|
| `--model` | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` | Same model as the baseline |
| `--train-type` | `lora` (rank 8, scale 10, all layers) | 15M trainable parameters; fits easily |
| `--group-size` | 4 | Answers per question. 8 wastes fewer groups (16% vs 36% all-same) but doubles generation |
| `--batch-size` | 4 | Questions per step (16 completions) |
| `--max-completion-length` | 256 | SQL is short: gold queries are at most 156 tokens, and sampled answers average about 35 |
| `--max-seq-length` | 1024 | Prompt (≤ 511 tokens) plus completion |
| `--temperature` | 1.0 | Variety within groups; same as the difficulty filter |
| `--learning-rate` | 5e-6, constant | Low, as is usual for RL fine-tuning |
| `--beta` | 0.04 | KL weight; small |
| `--grpo-loss-type` | `grpo` | `dr_grpo` is the planned second run |
| `--micro-batch-size` | 1 | Memory (section 9) |
| `--grad-checkpoint` | on | Memory |
| `--steps-per-report` / `--save-every` | 10 / 50 | |
| `--seed` | 0 | |
| `--iters` | set per run (600 for `grpo-v1`) | 600 × 4 = 2,400 questions sampled, about 0.75 of the 3,196 |

`train_grpo.py`'s own options:

| Option | Default | Purpose |
|---|---|---|
| `--run-name` | required | Output folder `models/<run>/` |
| `--input` | `data/rl/train_filtered.jsonl` | Training set |
| `--limit` | all | First N questions (smoke tests) |
| `--sample-every` | 50 | Log full answer groups to `rollouts.jsonl` every N steps |
| `--gpu-memory-gb` | 28 | MLX soft memory and wired limit |
| `--gpu-kill-gb` | 34 | Stop training if MLX memory passes this |

## 9. Memory on Apple Silicon

The first full run, with the loss computed on 4 completions at a time, **kernel-panicked the laptop** within minutes. The panic report showed the cause:

1. **Longer sequences than the smoke test.** Real prompts reach 511 tokens, against 287 in the smoke set. Most of the loss step's memory is logits over the 151,936-token vocabulary, which scale with the number and length of sequences processed at once.
2. **MLX pins most of the RAM.** mlx-lm's `BatchGenerator` raises the GPU wired-memory limit to the recommended 37.4 GB of 48 GB at every sampling round, and MLX's own memory limit defaults to 45.6 GB. Wired memory can't be swapped, so everything else was pushed into swap: 21 swap files.
3. **macOS stopped responding**, and `watchdogd` forced a panic and restart.

What changed:

| Change | Effect |
|---|---|
| `--micro-batch-size 1` | Logits for one completion at a time; same gradient |
| `--grad-checkpoint` | Activations recomputed instead of stored |
| [src/mlx_guard.py](src/mlx_guard.py) | Sets a soft memory and wired limit (28 GB) in place of mlx-lm's 37.4 GB. A watchdog thread exits the process (code 3) if MLX memory passes 34 GB. `mx.set_memory_limit` alone is only advisory: it never refuses an allocation |
| `caffeinate -i` | Keeps the Mac awake during training; the panic hit during a sleep/wake |
| `gpu_peak_gb` in `rollouts.jsonl` | Peak memory per step, to see trends before a crash |

After the change, peak memory is **6.7–8.0 GB** per step.

**LoRA rank doesn't matter for memory.** The adapter plus its gradients and Adam state take about 0.25 GB at rank 8, against several GB of logits per sequence. Rank is a quality setting.

**Run no other GPU job at the same time.** Training next to the difficulty filter, or next to another project's training job, crashed with Metal out-of-memory errors.

## 10. Reading the training output

### The trainer's report, every 10 steps

From the smoke test, before the memory changes (hence 17.9 GB):

```
Iter 20:
Loss: 0.002
Total Rewards:  μ=0.781, σ=0.352
Group Rewards:  μ=0.781, σ=0.147
KL Divergence: 0.060799174011
  • Avg tokens: 29.7   • Max tokens: 52 (limit: 256)   • Hit limit: 0.0%
Clipping:  low=0.000, high=0.000, total=0.000
Memory: 17.946GB
```

| Line | Meaning |
|---|---|
| Loss | About β × KL; not a progress measure (section 6) |
| Total Rewards μ | Mean reward over the last 10 steps: **the main progress signal** |
| Group Rewards σ | Mean within-group std; near 0 means most groups are all-same and teach nothing |
| KL Divergence | Drift from the base model; should grow slowly and stay small |
| Avg / Max tokens, Hit limit | Answer length; a steady rise means length bias |
| Clipping | Always 0 in this setup (on-policy) |
| Memory | Peak memory reported by the trainer |

### `rollouts.jsonl`, one line per step

```json
{"step": 350, "reward_mean": 0.8125, "all_same_groups": 0.75, "completion_chars": 114.2,
 "no_sql_block": 0.0, "gpu_peak_gb": 6.7, "samples": [...]}
```

`samples` appears on step 1 and every 50 steps: for 2 questions, the question, its gold SQL, and all 4 answers with their rewards. **Read them.** Reward hacking shows up in the samples before it shows up in the numbers.

### What healthy looks like

| Signal | Healthy | Problem |
|---|---|---|
| Mean reward | Rises slowly from about the filter's 0.51 | Flat: no learning. A sudden jump to 1.0: read the samples |
| All-same groups | 35–65% | Above 80%: most compute is wasted; use `--group-size 8` or re-filter |
| KL | Small (≪ 1), slowly rising or flat | Rising fast: lower the learning rate or raise β |
| Answer length | About 30–40 tokens, stable | Steady growth: length bias; try `dr_grpo` |
| Samples | Plain SQL in one code block | Odd patterns while reward rises: a hole in the reward |

## 11. Results so far

`grpo-v1`, as of step 421 of 600 (2026-10-03). Rewards are sampled at temperature 1.0 on training questions:

| Steps | Mean reward | All-same groups | Avg answer length | Peak memory |
|---|---|---|---|---|
| 1–10 | 0.56 | – | – | – |
| 1–50 | 0.65 | 56% | 114 chars | 7.2 GB |
| 101–150 | 0.70 | 65% | 114 | 6.8 GB |
| 201–250 | 0.74 | 67% | 121 | 8.0 GB |
| 301–350 | 0.80 | 78% | 114 | 6.7 GB |
| 351–400 | 0.77 | 78% | 120 | 7.5 GB |

- **The reward rose from about 0.56 to about 0.78–0.80**, on mostly unseen questions (about half the training set had been sampled by step 421). The first steps match the base model's 0.51 from the filter, sampled the same way.
- **No sign of reward hacking.** Samples are clean SQL with real variety among correct answers; length is stable; KL is flat at about 0.03–0.05.
- **All-same groups rose from 56% to 78%** as the model grew more confident; at step 350 all 4 answers to one question were identical. Expect learning to slow as more groups stop giving a signal.

**Not yet measured:** greedy accuracy on held-out databases, which is what counts. Compare checkpoints on val (baseline 59.5%), then report the chosen one on test (baseline 66.0%), with [step 7](#step-7-evaluate).

## 12. mlx-lm-lora pitfalls

| Pitfall | Handling |
|---|---|
| **PyPI 3.1.3 has a broken GRPO loss.** It computes log-probabilities of the completion **without the prompt**, and clips against the frozen reference model with ε = 1e-4, which stalls learning after the first tiny move | Pinned to GitHub commit `4600922` (3.4.3) in `pyproject.toml`, which fixes both |
| **Default system prompt.** Without a `system` field, rows get an R1-style "think in `<think>` tags, answer in `<answer>` tags" prompt | `train_grpo.py` sets Qwen's default system prompt; training prompts are token-identical to `eval_sql.py`'s (200 of 200 checked) |
| **Auto-fuse.** After training it writes a de-quantized 5.8 GB model into the adapter folder, and `--fuse` can only switch that on | `train_grpo.py` passes a config file with `fuse: false` |
| **Built-in validation** reports only GRPO loss, which says nothing about accuracy | No validation file; checkpoints are scored with `eval_sql.py` |
| **`</answer>` stop token** left over from the R1 default | Harmless for SQL |

## 13. Known limitations

- **Spider labels have errors.** Some gold SQL doesn't answer its question (e.g. `test-5` selects 2 of the 3 requested columns). Correct answers to those score 0, for every model equally.
- **Execution match on one database instance** can be fooled when different queries return the same rows (0.5–1.4% of rows; [section 3](#3-the-reward)). Spider's "test-suite" evaluation, which runs on several database variants, would be stricter.
- **The train-set ceiling is optimistic**: the retriever was trained on train questions, so retrieval is easier there than on val/test.
- **Multi-database prompts** are realistic for retrieval over all 876 tables, but they cause about 9% of baseline failures on test (tables from the wrong database). A setup where the database is known (`--pool db`) would remove that, and also most of the retriever's role.
- **Group size 4** loses many groups to identical rewards (section 5), increasingly so as the model improves.
- **`dr_grpo`** here changes only the token averaging, not the std normalization of advantages that the Dr. GRPO paper also removes.

---

## Usage

```bash
uv run python src/sql_exec.py                                                    # 1. databases
uv run python src/build_rl_prompts.py                                            # 2. prompts
uv run python src/eval_sql.py --split test                                       # 3. baseline (also --split val)
uv run python src/filter_difficulty.py                                           # 4. difficulty filter (~1 h)
uv run python src/train_grpo.py --run-name grpo-smoke --limit 50 --iters 20      # 5. smoke test
uv run python src/train_grpo.py --run-name grpo-v1 --iters 600                   # 6. train (~2 h)
uv run python src/eval_sql.py --split val --adapter models/grpo-v1/adapters --tag grpo-v1   # 7. evaluate
```
