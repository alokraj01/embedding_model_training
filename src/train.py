"""Fine-tune bge for question -> table retrieval.

Loss: MultipleNegativesRankingLoss on (question, gold table, same-database hard negative) triplets,
wrapped in MatryoshkaLoss so truncated embeddings also work.

Every epoch it records:
  train_objective  the loss actually optimised, averaged over the epoch's training steps
  train/val pool loss
                   cross-entropy of each gold table against all 876 tables (other gold tables of the
                   same question excluded), at full size. Computed the same way for val questions and
                   a fixed sample of train questions, so the two curves are directly comparable.
  val recall@k     each val question ranked against all 876 tables, at full and truncated sizes
Epoch 0 is the untrained model. The epoch with the best val recall@5 (full size) is kept as the final model.

Why not the trainer's usual eval loss: MNRL scores each question against the other tables in its batch,
and batches cannot repeat a table. Val has only 60 tables (train has 677), so val batches hold far fewer
candidates and its batch loss is on a different, lower scale than train loss.

Outputs in models/<run-name>/:
  final/               best model (load with SentenceTransformer("models/<run-name>/final"))
  history.csv          one row per epoch
  training_curves.png

Usage:
  uv run python src/train.py                                    # bge-small, 5 epochs
  uv run python src/train.py --model BAAI/bge-base-en-v1.5 --batch-size 32
  uv run python src/train.py --no-hard-negatives                # ablation
  uv run python src/train.py --mask-gold                        # don't score other gold tables as wrong
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
    SentenceEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from transformers import TrainerCallback

from evaluate import BGE_QUERY_PROMPT, DIMS, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MAX_NEGS_PER_PAIR = 3
MNRL_SCALE = 20.0  # MultipleNegativesRankingLoss default (temperature 0.05)
TRAIN_SAMPLE_QUESTIONS = 1000  # train questions used for the train pool loss


def build_rows(split, hard_negatives, seed, limit=None):
    """(anchor, positive[, negative]) rows from {split}_pairs.jsonl, shuffled so batches mix databases.

    Returns (dataset, meta), where meta[i] = (qid, [positive table id, negative table id]) for dataset row i.
    """
    rng = random.Random(seed)
    rows = []
    for p in read_jsonl(DATA / f"{split}_pairs.jsonl"):
        if not hard_negatives:
            rows.append(({"anchor": p["anchor"], "positive": p["positive"]}, (p["qid"], [p["positive_id"]])))
            continue
        negs = list(zip(p["hard_negatives"], p["hard_negative_ids"]))
        for neg, neg_id in rng.sample(negs, min(MAX_NEGS_PER_PAIR, len(negs))):
            rows.append(({"anchor": p["anchor"], "positive": p["positive"], "negative": neg},
                         (p["qid"], [p["positive_id"], neg_id])))
    rng.shuffle(rows)
    rows = rows[:limit] if limit else rows
    return Dataset.from_list([r for r, _ in rows]), [m for _, m in rows]


class MaskedGoldMNRL(MultipleNegativesRankingLoss):
    """MultipleNegativesRankingLoss that never scores a question's other gold tables as wrong.

    A 3-table question is trained as 3 rows, one per gold table. Plain MNRL treats every other candidate in
    the batch as wrong, so when another row brings in one of the question's other gold tables (as its
    positive or hard negative), that table is pushed away; this happens in about a third of multi-table rows.
    Here those candidates are removed from the row's softmax, as in PoolLossEvaluator; the row's own positive
    stays the target.

    Each row's dataset index arrives as its `label`. Only the default MNRL setup is supported
    (question -> table direction, single device).
    """

    def __init__(self, model, row_gold, row_candidates, scale=MNRL_SCALE):
        super().__init__(model, scale=scale)
        self.row_gold = row_gold  # dataset row -> set of the question's gold table ids
        self.row_candidates = row_candidates  # dataset row -> [positive id, negative id]

    def compute_loss_from_embeddings(self, embeddings, labels):
        rows = labels.tolist()
        n = len(rows)
        # Candidates in the same order as the embeddings: every row's positive, then every row's negative.
        cand_ids = [self.row_candidates[r][c] for c in range(len(embeddings) - 1) for r in rows]
        mask = torch.tensor([[cid in self.row_gold[r] for cid in cand_ids] for r in rows])
        mask[range(n), range(n)] = False  # the row's own positive is the target

        scores = self.scale * self.similarity_fct(embeddings[0], torch.cat(embeddings[1:]))
        scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
        return F.cross_entropy(scores, torch.arange(n, device=scores.device))


class PoolLossEvaluator(SentenceEvaluator):
    """Mean cross-entropy of each (question, gold table) against every table in the pool.

    Uses the same scale as MNRL, but a fixed candidate set, so the value is comparable across
    splits and epochs. Other gold tables of the same question are masked out, not counted as wrong.
    """

    def __init__(self, queries, tables, name, scale=MNRL_SCALE, batch_size=64):
        super().__init__()
        self.queries, self.name, self.scale, self.batch_size = queries, name, scale, batch_size
        self.table_texts = [t["text"] for t in tables]
        idx = {t["table_id"]: i for i, t in enumerate(tables)}
        self.gold = [[idx[g] for g in q["gold_table_ids"]] for q in queries]
        self.greater_is_better = False
        self.primary_metric = "pool_loss"

    @torch.no_grad()
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        enc = dict(batch_size=self.batch_size, convert_to_tensor=True, normalize_embeddings=True)
        q = model.encode([BGE_QUERY_PROMPT + x["question"] for x in self.queries], **enc)
        t = model.encode(self.table_texts, **enc)
        logits = self.scale * (q @ t.T)
        losses = []
        for i, gold in enumerate(self.gold):
            for g in gold:
                row = logits[i].clone()
                row[[o for o in gold if o != g]] = float("-inf")
                losses.append(F.cross_entropy(row.unsqueeze(0), torch.tensor([g], device=row.device)))
        metrics = self.prefix_name_to_metrics({"pool_loss": torch.stack(losses).mean().item()}, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics


def build_evaluator(dims, seed, limit=None):
    tables = read_jsonl(DATA / "tables.jsonl")
    val_q = read_jsonl(DATA / "val_queries.jsonl")[:limit]
    train_q = read_jsonl(DATA / "train_queries.jsonl")
    train_q = random.Random(seed).sample(train_q, min(limit or TRAIN_SAMPLE_QUESTIONS, len(train_q)))

    recall = [
        InformationRetrievalEvaluator(
            queries={q["qid"]: q["question"] for q in val_q},
            corpus={t["table_id"]: t["text"] for t in tables},
            relevant_docs={q["qid"]: set(q["gold_table_ids"]) for q in val_q},
            query_prompt=BGE_QUERY_PROMPT,
            precision_recall_at_k=[1, 3, 5, 10],
            accuracy_at_k=[1, 3, 5],
            truncate_dim=d,
            name=f"val876_dim{d}",
            batch_size=64,
        )
        for d in dims
    ]
    pool = [PoolLossEvaluator(train_q, tables, "train"), PoolLossEvaluator(val_q, tables, "val")]
    # The first evaluator (val recall at full size) is the score used to pick the best epoch.
    return SequentialEvaluator(recall + pool, main_score_function=lambda scores: scores[0])


class EpochHistory(TrainerCallback):
    """Collect per-epoch training objective and evaluator metrics from the trainer's logs."""

    def __init__(self):
        self.rows = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        epoch = round(logs.get("epoch", state.epoch or 0))
        row = self.rows.setdefault(epoch, {"epoch": epoch})
        if "loss" in logs:
            row["train_objective"] = logs["loss"]
        for k, v in logs.items():
            if k.startswith("eval_") and isinstance(v, (int, float)):
                row[k.removeprefix("eval_")] = v

    def table(self):
        return [self.rows[e] for e in sorted(self.rows)]


class ReleaseMPSCache(TrainerCallback):
    """Return cached MPS memory after every step.

    Batches are padded to their own longest sequence, so every step allocates differently shaped
    tensors. The MPS allocator keeps all of them cached; without this a bge-small run grew to 47 GB
    of GPU memory, pushed the machine into swap and slowed from ~2 s to ~14 s per step.
    """

    def on_step_end(self, args, state, control, **kwargs):
        torch.mps.empty_cache()

    on_evaluate = on_step_end


def save_history(rows, out_dir, dims):
    keys = ["epoch", "train_objective", "train_pool_loss", "val_pool_loss"] + [
        f"val876_dim{d}_cosine_recall@{k}" for d in dims for k in (3, 5)
    ]
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])

    def series(key):
        return list(zip(*[(r["epoch"], r[key]) for r in rows if key in r])) or [[], []]

    fig, (ax_obj, ax_pool, ax_rec) = plt.subplots(1, 3, figsize=(17, 4.6))
    ax_obj.plot(*series("train_objective"), "o-", color="tab:blue")
    ax_obj.set(title="Training objective\n(Matryoshka-summed MNRL, avg over epoch)", xlabel="epoch", ylabel="loss")

    ax_pool.plot(*series("train_pool_loss"), "o-", label="train (1,000 sampled questions)")
    ax_pool.plot(*series("val_pool_loss"), "s-", label="val (unseen databases)")
    ax_pool.set(title=f"Pool loss: gold table vs all 876 tables\n({dims[0]} dims, comparable across splits)",
                xlabel="epoch", ylabel="cross-entropy")
    ax_pool.legend()

    for d, style in ((dims[0], "-"), (dims[-1], "--")):
        for k, marker in ((3, "o"), (5, "s")):
            ax_rec.plot(*series(f"val876_dim{d}_cosine_recall@{k}"), marker + style, label=f"recall@{k}, {d} dims")
    ax_rec.set(title="Val recall (876-table pool)", xlabel="epoch", ylabel="recall")
    ax_rec.legend()

    for ax in (ax_obj, ax_pool, ax_rec):
        ax.grid(alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    print(f"saved {out_dir / 'history.csv'} and {out_dir / 'training_curves.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--run-name", default=None, help="default: <model>-mnrl[-matryoshka][-no-hn]")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64, help="larger = more in-batch negatives")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=256, help="longest table description is 295 tokens")
    p.add_argument("--no-hard-negatives", action="store_true", help="ablation: (question, table) pairs only")
    p.add_argument("--no-matryoshka", action="store_true", help="plain MNRL at full size only")
    p.add_argument("--mask-gold", action="store_true",
                   help="never score a question's other gold tables as wrong (MaskedGoldMNRL)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="use only N rows/questions (smoke test)")
    args = p.parse_args()

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length
    full = model.get_embedding_dimension()
    dims = [full] + [d for d in DIMS if d < full]

    hard_negs = not args.no_hard_negatives
    run_name = args.run_name or (
        args.model.split("/")[-1].replace("-en-v1.5", "")
        + "-mnrl" + ("" if args.no_matryoshka else "-matryoshka") + ("" if hard_negs else "-no-hn")
        + ("-maskgold" if args.mask_gold else "")
    )
    out_dir = MODELS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, row_meta = build_rows("train", hard_negs, args.seed, args.limit)
    if args.mask_gold:
        gold = {q["qid"]: set(q["gold_table_ids"]) for q in read_jsonl(DATA / "train_queries.jsonl")}
        # The collator passes a "label" column to the loss; here it is the row index into row_meta.
        train_ds = train_ds.add_column("label", list(range(len(train_ds))))
        loss = MaskedGoldMNRL(model, [gold[qid] for qid, _ in row_meta], [ids for _, ids in row_meta])
    else:
        loss = MultipleNegativesRankingLoss(model, scale=MNRL_SCALE)
    print(f"train rows: {len(train_ds)}  columns: {train_ds.column_names}")

    if not args.no_matryoshka:
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=dims)

    evaluator = build_evaluator(dims, args.seed, args.limit)
    best_metric = f"eval_val876_dim{full}_cosine_recall@5"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_ratio,  # float < 1 = fraction of total steps (transformers v5)
        # The same table is the positive for many questions; without this, a batch can contain a
        # question's own gold table as someone else's "negative".
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Questions get the bge retrieval instruction, tables are embedded as-is (same as evaluate.py).
        prompts={"anchor": BGE_QUERY_PROMPT},
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        dataloader_pin_memory=False,  # not supported on MPS
        seed=args.seed,
        report_to="none",
        run_name=run_name,
    )

    history = EpochHistory()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
        callbacks=[history] + ([ReleaseMPSCache()] if torch.backends.mps.is_available() else []),
    )

    # Epoch 0: the untrained model, so the curves start from the baseline.
    trainer.log({**trainer.evaluate(), "epoch": 0})
    trainer.train()

    rows = history.table()
    save_history(rows, out_dir, dims)
    (out_dir / "run_config.json").write_text(json.dumps({**vars(args), "run_name": run_name, "dims": dims,
                                                          "best_metric": best_metric}, indent=2))
    model.save(str(out_dir / "final"))

    cols = [("train_objective", "objective"), ("train_pool_loss", "train pool"), ("val_pool_loss", "val pool"),
            (f"val876_dim{full}_cosine_recall@3", "val R@3"), (f"val876_dim{full}_cosine_recall@5", "val R@5"),
            (f"val876_dim{dims[-1]}_cosine_recall@5", f"R@5 {dims[-1]}d")]
    print("\n" + f"{'epoch':>5} " + " ".join(f"{label:>11}" for _, label in cols))
    for r in rows:
        print(f"{r['epoch']:>5} " + " ".join(f"{r[k]:>11.4f}" if k in r else f"{'-':>11}" for k, _ in cols))
    print(f"\nbest model (by val recall@5, {full} dims) saved to {(out_dir / 'final').relative_to(ROOT)}")
    print(f"test it: uv run python src/evaluate.py --model {(out_dir / 'final').relative_to(ROOT)} "
          f"--split test --pool all --tag {run_name}")


if __name__ == "__main__":
    main()
