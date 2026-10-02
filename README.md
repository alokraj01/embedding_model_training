# Table retrieval embeddings

Fine-tunes [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (or `bge-base`) with [sentence-transformers](https://www.sbert.net/) so that a natural-language question retrieves the database tables its SQL needs. This is the schema-linking step of a text-to-SQL pipeline: with thousands of tables you can't put every schema in the prompt, so you first retrieve the few that matter.

```
"How many classes does professor Graztevski teach?"
        │
        ▼  embed, nearest-neighbour search over all table vectors
  1. college_1.class      ✓
  2. college_1.employee   ✓
  3. college_1.course
```

Training data comes from [Spider](https://yale-lily.github.io/spider). The model is evaluated only on **databases it never saw during training**.

## Results

Spider dev (20 unseen databases, 1,034 questions). Each question ranks **all 876 tables** across every Spider database.

| Model | Dim | recall@3 | recall@5 | complete@5 | MRR |
|---|---|---|---|---|---|
| bge-base-en-v1.5 (no fine-tuning) | 768 | 0.794 | 0.873 | 0.789 | 0.817 |
| bge-small-en-v1.5 (no fine-tuning) | 384 | 0.818 | 0.883 | 0.799 | 0.834 |
| **bge-small, fine-tuned** | 384 | **0.830** | **0.887** | **0.802** | **0.856** |

- **recall@k:** share of a question's gold tables found in the top *k*.
- **complete@k:** share of questions with *every* gold table in the top *k*, which is what a downstream SQL generator actually needs.

### Smaller vectors

The fine-tuned model is trained with Matryoshka loss, so you can cut its vectors down and still get good results. The off-the-shelf model falls apart when you do:

| Dim | bge-small (no fine-tuning) recall@5 | fine-tuned recall@5 |
|---|---|---|
| 384 | 0.883 | 0.887 |
| 256 | 0.857 | 0.883 |
| 128 | 0.771 | 0.867 |
| 64 | 0.597 | **0.833** |

At full size, fine-tuning helps only a little (+1.2 points recall@3, +2.2 MRR). The main gain is that a **64-dim vector, 6× smaller**, keeps 94% of full-size recall@5. Without fine-tuning it keeps 68%.

### Multi-table questions are the hard part

About 45% of questions need two or more tables. Fine-tuned bge-small, 384 dims:

| Gold tables | Questions | recall@5 | complete@5 |
|---|---|---|---|
| 1 | 575 | 0.965 | 0.965 |
| 2 | 393 | 0.816 | 0.654 |
| 3+ | 66 | 0.635 | 0.258 |

## How it works

1. **Build the dataset** ([src/build_dataset.py](src/build_dataset.py)). Each question's gold SQL is parsed with [sqlglot](https://github.com/tobymao/sqlglot), subqueries included, to find the tables it uses. Every table becomes a short text description:

   ```
   Table: CLASS
   Columns: class code (text, primary key), course code (text), class section (text), ...
   Foreign keys: PROF_NUM -> EMPLOYEE.EMP_NUM; CRS_CODE -> COURSE.CRS_CODE
   ```

   Splits are by **database**, not by question, so test scores measure generalisation to new schemas:

   | Split | Source | Databases | Tables | Questions |
   |---|---|---|---|---|
   | train | Spider train, 90% of databases | 126 | 677 | 6,366 |
   | val | Spider train, the other 10% | 14 | 60 | 625 |
   | test | Spider dev | 20 | 81 | 1,034 |

2. **Train** ([src/train.py](src/train.py)) on `(question, gold table, hard negative)` triplets, where the hard negative is another table from the **same database**. Same-database tables share vocabulary (`CLASS`, `COURSE`, `ENROLL`), so they are the confusions the model has to learn to resolve. The loss is `MultipleNegativesRankingLoss` (InfoNCE with in-batch negatives) wrapped in `MatryoshkaLoss` at 384/256/128/64 dims. The epoch with the best val recall@5 is kept.

3. **Evaluate** ([src/evaluate.py](src/evaluate.py)) by ranking each question against a pool of tables. It reports recall@k, complete@k and MRR at every Matryoshka size, broken down by the number of gold tables.

[02_training_and_loss.md](02_training_and_loss.md) walks through the loss with a worked example. It also covers why hard negatives matter, the batching rules, and how to read the training curves.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Training runs on Apple Silicon (MPS), CUDA or CPU.

```bash
uv sync

# Build data/processed/ (already committed; downloads Spider via HF datasets)
uv run python src/build_dataset.py

# Smoke test, ~1 min
uv run python src/train.py --limit 200 --epochs 1 --run-name _smoke

# Full run: bge-small, 5 epochs, hard negatives + Matryoshka
uv run python src/train.py --run-name bge-small-mnrl-matryoshka

# Evaluate the fine-tuned model and the baseline on the test databases
uv run python src/evaluate.py --model models/bge-small-mnrl-matryoshka/final --split test --pool all --tag bge-small-mnrl-matryoshka
uv run python src/evaluate.py --model BAAI/bge-small-en-v1.5 --split test --pool all
```

Training writes to `models/<run-name>/`: the best model in `final/`, per-epoch metrics in `history.csv`, plots in `training_curves.png`, and the arguments in `run_config.json`. Evaluation writes `results/<tag>__<split>__<pool>.json`. Both folders are gitignored.

### Options

| Flag | Effect |
|---|---|
| `--model BAAI/bge-base-en-v1.5` | larger base model (use `--batch-size 32` on a laptop) |
| `--no-hard-negatives` | ablation: train on (question, table) pairs only |
| `--no-matryoshka` | ablation: train at full size only |
| `--mask-gold` | don't count a question's *other* gold tables as wrong answers in the loss |
| `--pool split` (evaluate) | rank only the split's own tables instead of all 876 |

## Using the model

Questions need the bge query prefix; table descriptions don't.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/bge-small-mnrl-matryoshka/final", truncate_dim=128)

query = model.encode(
    "Represent this sentence for searching relevant passages: How many singers are from France?",
    normalize_embeddings=True,
)
tables = model.encode(table_descriptions, normalize_embeddings=True)
scores = tables @ query
```

## Project layout

```
data/processed/        tables.jsonl, {train,val,test}_{pairs,queries}.jsonl, stats.json
src/build_dataset.py   Spider → table descriptions + gold tables per question
src/train.py           fine-tuning (MNRL + Matryoshka, hard negatives)
src/evaluate.py        retrieval metrics against a table pool
02_training_and_loss.md  in-depth explanation of the training setup
```

## Limitations

- Spider schemas are small and clean, with about 5 tables per database on average. Real warehouses with hundreds of cryptically named tables will be harder.
- `--mask-gold` did not measurably change test results in this setup (recall@5 0.884 vs 0.887).
- No API embedding model has been compared yet.
