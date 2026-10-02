# How training and the loss function work

This project fine-tunes a `bge` embedding model so that a natural-language question lands close to the database tables needed to answer it. This document explains what one training step does, how the loss is calculated, and how to read the numbers the training script prints. Everything here describes [src/train.py](src/train.py) with sentence-transformers 6.1.

- [1. The task](#1-the-task)
- [2. Training data](#2-training-data)
- [3. One training step, end to end](#3-one-training-step-end-to-end)
- [4. MultipleNegativesRankingLoss](#4-multiplenegativesrankingloss)
- [5. Why hard negatives matter](#5-why-hard-negatives-matter)
- [6. MatryoshkaLoss](#6-matryoshkaloss)
- [7. Batching rules](#7-batching-rules)
- [8. The training loop](#8-the-training-loop)
- [9. Reading the training output](#9-reading-the-training-output)
- [10. Running on Apple Silicon](#10-running-on-apple-silicon)
- [11. Known limitations](#11-known-limitations)

---

## 1. The task

Given a question such as *"How many classes does professor Graztevski teach?"*, find the tables a SQL query would need (`CLASS`, `EMPLOYEE`) out of every table we know about.

The model is a **bi-encoder**: questions and tables are each turned into a vector on their own, and relevance is the cosine similarity between the two vectors. Table vectors can be computed once and stored, so retrieval is a single nearest-neighbour search.

Fine-tuning changes the model's weights so that, in that vector space:

- a question moves **closer** to the tables its SQL uses, and
- it moves **further** from tables it doesn't use, especially look-alike tables from the same database.

## 2. Training data

[src/build_dataset.py](src/build_dataset.py) builds everything from [Spider](https://yale-lily.github.io/spider), a text-to-SQL dataset. It parses each question's gold SQL to find which tables it uses.

### Each table becomes a text description

```
Table: CLASS
Columns: class code (text, primary key), course code (text), class section (text),
         class time (text), class room (text), professor employee number (number)
Foreign keys: PROF_NUM -> EMPLOYEE.EMP_NUM; CRS_CODE -> COURSE.CRS_CODE
```

### Splits are by database, not by question

| Split | Source | Databases | Tables | Questions |
|---|---|---|---|---|
| train | Spider train, 90% of databases | 126 | 677 | 6,366 |
| val | Spider train, the other 10% | 14 | 60 | 625 |
| test | Spider dev | 20 | 81 | 1,034 |

Val and test databases never appear in training, so the scores measure how well the model handles **schemas it has never seen**. Splitting by question would let the model memorise table names instead.

### Training rows are triplets

A question that uses 2 tables produces 2 (question, table) **pairs**. Each pair then produces up to 3 **triplets**, one per randomly chosen hard negative:

| Column | Role | Example |
|---|---|---|
| `anchor` | the question | *"How many classes are professor whose last name is Graztevski has?"* |
| `positive` | a table the gold SQL uses | `Table: CLASS …` |
| `negative` | a table from the **same database** that the SQL does **not** use | `Table: COURSE …` |

That gives 9,917 pairs and 21,075 triplets. A table used by the question is never picked as one of its negatives, even when the row's positive is a different table.

Running with `--no-hard-negatives` drops the `negative` column and trains on pairs alone, to measure how much the hard negatives add.

## 3. One training step, end to end

With the defaults (bge-small, batch size 64):

```
64 triplets
   │
   ├─ anchors    ─→ "Represent this sentence for searching relevant passages: " + question
   ├─ positives  ─→ table text, no prefix
   └─ negatives  ─→ table text, no prefix
   │
   ▼  encode each column with the model
Q (64 × 384)   P (64 × 384)   N (64 × 384)
   │
   ▼  MatryoshkaLoss: for each size d in [384, 256, 128, 64]
   │     keep the first d numbers of every vector, re-normalize to length 1
   │     run MultipleNegativesRankingLoss on the truncated vectors
   │
   ▼  add the 4 losses together
one number → backpropagate → AdamW updates the weights
```

The query prefix is the instruction bge v1.5 was trained with for short-query retrieval. It's applied through `prompts={"anchor": BGE_QUERY_PROMPT}` in [train.py](src/train.py), so the `anchor` column name matters. [src/evaluate.py](src/evaluate.py) applies the same prefix at test time.

## 4. MultipleNegativesRankingLoss

MNRL (also called InfoNCE) turns retrieval into a classification problem: *out of every table in this batch, which one is mine?*

### Build the similarity matrix

Stack the positives and negatives into one list of 128 candidates, then score every question against every candidate:

```
S = 20 × cos(Q, [P ; N])          shape 64 × 128
```

```
                 positives                hard negatives
              P1    P2   …   P64    │   N1    N2   …   N64
    q1     [ ✓ ]    ·         ·     │   ·     ·         ·
    q2       ·    [ ✓ ]       ·     │   ·     ·         ·
    …
    q64      ·      ·       [ ✓ ]   │   ·     ·         ·
```

For question *i*, the correct answer is column *i*, its own positive. The other 127 columns all count as wrong:

- 63 positives belonging to other questions (**in-batch negatives**, mostly from other databases),
- its own hard negative,
- 63 hard negatives belonging to other questions.

The factor 20 is the **scale**, equivalent to a softmax temperature of 0.05. Cosine similarity only ranges from −1 to 1; scaling it up lets the softmax become confident.

### Cross-entropy on each row

```
loss_i = −log( exp(S[i,i]) / Σ_j exp(S[i,j]) )

MNRL   = average of loss_i over the 64 questions
```

This is ordinary softmax cross-entropy where the label for row *i* is *i*.

### Worked example

One question, four candidates instead of 128:

| Candidate | cosine | × 20 | softmax |
|---|---|---|---|
| own positive (`CLASS`) | 0.80 | 16 | **0.881** |
| own hard negative (`COURSE`) | 0.70 | 14 | 0.119 |
| another question's table | 0.30 | 6 | 0.00004 |
| another question's hard negative | 0.20 | 4 | 0.000005 |

```
loss = −ln(0.881) ≈ 0.127
```

### What the gradient does

The gradient of the loss with respect to each score is:

```
∂loss / ∂S[i,j] = softmax_j − (1 if j is the positive else 0)
```

So the positive is pulled up by `1 − 0.881 = 0.119`, and **each wrong candidate is pushed down by exactly its softmax probability**. In the example, `COURSE` receives almost all of the push (0.119) while the two easy tables receive close to nothing. The model learns mostly from the candidates it currently finds confusing.

## 5. Why hard negatives matter

Without the `negative` column, every wrong answer in the batch is some other question's table. With 126 databases, those are almost always from a different schema: telling *"singer"* apart from *"flight"* is easy, and the loss quickly approaches zero with little left to learn.

The real retrieval problem is harder. A database's tables share vocabulary: `CLASS`, `COURSE` and `ENROLL` all mention courses. Hard negatives put exactly these near-misses into every question's softmax. As section 4 shows, the near-misses are where the gradient goes, so the model is forced to learn which columns actually answer the question.

Comparing a default run with a `--no-hard-negatives` run measures this directly.

## 6. MatryoshkaLoss

A normally trained embedding spreads its information across all of its dimensions, so cutting a 384-number vector down to 64 numbers ruins it. MatryoshkaLoss trains the model so that **every prefix of the vector is a usable embedding on its own**, with the most important information in the first dimensions.

### How it's calculated

The model runs once. MatryoshkaLoss then repeats MNRL on progressively shorter versions of the same embeddings and adds the results:

```
total = MNRL(first 384 dims) + MNRL(first 256) + MNRL(first 128) + MNRL(first 64)
```

- Each truncated vector is **re-normalized to length 1** before scoring, so cosine similarity stays valid.
- Every size has weight 1, and the losses are **summed**, not averaged.
- The sizes are the model's full size plus every smaller entry in `DIMS` ([src/evaluate.py](src/evaluate.py)):

| Model | Full size | Sizes trained |
|---|---|---|
| bge-small-en-v1.5 | 384 | 384, 256, 128, 64 |
| bge-base-en-v1.5 | 768 | 768, 512, 384, 256, 128, 64 |

### Why bother

A 64-dimension vector is 6× smaller than a 384-dimension one, so the index is smaller and search is faster. A Matryoshka-trained model typically loses only a little recall when truncated, while an ordinary model degrades sharply. The evaluation reports recall at every size so the two can be compared.

At inference time, pick the size when loading the model:

```python
SentenceTransformer("models/<run-name>/final", truncate_dim=128)
```

## 7. Batching rules

### No duplicate texts within a batch

The same table is the positive for many questions; a popular table like `singer` might be the positive for dozens of rows. If two of those rows landed in the same batch, the loss would score `singer` as a **wrong** answer for one of them, even though it's correct.

`batch_sampler=BatchSamplers.NO_DUPLICATES` prevents this. It builds batches in which no text appears twice in **any** column, so a table can't be one row's positive and another row's negative, and the same question can't appear twice with different negatives.

### Batch size

A bigger batch means more candidates in each softmax, and generally a better model. With batch size *B* and hard negatives, each question is scored against *2B* candidates (128 at the default of 64). The cost is memory: every row's three texts sit in GPU memory at once.

If memory is the limit, `CachedMultipleNegativesRankingLoss` computes the same loss for a large batch in smaller chunks, trading extra compute time for much lower memory.

## 8. The training loop

| Setting | Default | Why |
|---|---|---|
| model | `BAAI/bge-small-en-v1.5` | fast on a laptop; `--model BAAI/bge-base-en-v1.5` for the larger one |
| epochs | 5 | |
| batch size | 64 | 128 candidates per question |
| learning rate | 2e-5 | standard for fine-tuning a pretrained encoder |
| warmup | first 10% of steps | avoids large updates while the optimizer's statistics settle |
| max sequence length | 256 tokens | the longest table description is 295 tokens; nearly all fit |
| steps | 330 per epoch, 1,650 total | 21,075 rows ÷ 64 |

At the end of every epoch the script evaluates on val, saves a checkpoint, and records the metrics. The **epoch with the best val recall@5 at full size** is kept as the final model, so later epochs that start to overfit don't overwrite a better one.

Outputs in `models/<run-name>/`:

- `final/`: the best model
- `history.csv`: one row of metrics per epoch
- `training_curves.png`: plots of the three measurements below
- `run_config.json`: the arguments used

## 9. Reading the training output

Each epoch records three kinds of numbers. Epoch 0 is the untrained model, so every curve starts from the baseline.

### `train_objective`

The loss actually being minimised, averaged over the epoch's steps. Because it's a **sum over 4 or 6 sizes**, it's larger than a single MNRL value. A model guessing at random scores about:

```
per size:   ln(128) ≈ 4.85        (ln(64) ≈ 4.16 with --no-hard-negatives)
bge-small:  4 sizes × 4.85 ≈ 19.4
bge-base:   6 sizes × 4.85 ≈ 29.1
```

Divide the logged value by the number of sizes to compare it with the per-size baseline.

### `train_pool_loss` and `val_pool_loss`

The same cross-entropy as MNRL, but each question is scored against **all 876 tables** instead of the 128 candidates in its batch. The question's other gold tables are left out of the softmax rather than counted as wrong. Random guessing scores about `ln(876) ≈ 6.78`.

These exist because the trainer's usual eval loss isn't comparable between splits: batches can't repeat a table, and val has only 60 tables to train's 677, so val batches have far fewer candidates. Scoring both splits against the same fixed pool of 876 tables makes the two curves directly comparable:

- both falling: the model is learning something general,
- train falling while val rises: the model is overfitting to the training schemas.

### Val recall@k

Each val question ranks all 876 tables. **recall@k** is the share of its gold tables that appear in the top *k*. It's reported at every Matryoshka size; the plot shows recall@3 and recall@5 at the full and the smallest size. This is the number that picks the best epoch.

### Final test

After training, evaluate the best model on the test databases:

```bash
uv run python src/evaluate.py --model models/<run-name>/final --split test --pool all --tag <run-name>
```

`--pool split` ranks against only the 81 test tables; `--pool all` ranks against every table and is the harder setting.

## 10. Running on Apple Silicon

Training runs on the Mac GPU through PyTorch's MPS backend. Each batch is padded to the length of its own longest text, so every step allocates differently sized tensors, and the MPS allocator keeps all of them cached instead of reusing them. In one bge-small run, GPU memory grew to 47 GB, the machine started swapping, and steps slowed from about 2 s to about 14 s.

The `ReleaseMPSCache` callback in [train.py](src/train.py) frees that cache after every step and every evaluation. If a run still slows down part-way through, check its memory:

```bash
footprint $(pgrep -f "src/train.py" | tail -1) | head -2
```

## 11. Known limitations

- **Pairs without hard negatives are dropped.** 1,212 of the 9,917 training pairs come from databases where every table is used by the question (often single-table databases), so they have no hard negative. Triplet rows can't be built for them, so in the default mode those pairs aren't trained on at all.
- **A question's other gold tables are often scored as wrong.** `NO_DUPLICATES` stops the *same* text appearing twice, but not a different gold table. If a question uses `singer` and `concert`, and another row in the same batch has `concert` as its positive or hard negative, `concert` is scored as wrong for the first question. Simulating one epoch of the real batches (batch size 64) found this in **35% of multi-table rows**: 26% of rows for 2-table questions, 50% for 3-table questions and 81% for 4-table questions. Single-table questions are unaffected. `--mask-gold` fixes this with `MaskedGoldMNRL`, which removes each question's other gold tables from its softmax (the same masking `PoolLossEvaluator` uses), while the row's own positive stays the target. Each row carries its dataset index as a `label` column so the loss can look up the question's gold tables. Compare runs with and without it using the "by number of gold tables" breakdown that [src/evaluate.py](src/evaluate.py) prints.
- **Each pair appears three times per epoch** (once per hard negative). This multiplies the training time by three compared with using one hard negative per pair and picking a fresh one each epoch.

## Usage

```bash
uv run python src/build_dataset.py                       # build data/processed/
uv run python src/train.py                               # bge-small, 5 epochs
uv run python src/train.py --model BAAI/bge-base-en-v1.5 --batch-size 32
uv run python src/train.py --no-hard-negatives           # ablation: pairs only
uv run python src/train.py --no-matryoshka               # ablation: full size only
uv run python src/train.py --mask-gold                   # don't score a question's other gold tables as wrong
uv run python src/train.py --limit 200 --epochs 1        # quick smoke test
```
