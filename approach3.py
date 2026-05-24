# =============================================================================
# approach3.py  —  Corrected version
# =============================================================================
#
# WHAT CHANGED FROM THE ORIGINAL AND WHY:
#
#   Original design flaw:
#     The original script claimed to ablate on BPE vocabulary size (8k/16k/32k)
#     by training three separate SentencePiece models. However, step4_finetune()
#     always loaded MarianTokenizer.from_pretrained(MODEL_NAME) — the same
#     fixed tokenizer every run. The custom SPM models were trained but never
#     applied, so all three runs were identical → identical BLEU scores.
#
#     Even if the SPM swap had been coded correctly, it cannot work with
#     MarianMT: the model's embedding layer is a fixed-size weight matrix
#     trained alongside its original vocabulary. Swapping in a different SPM
#     model would produce token IDs the model has never seen, destroying
#     translation quality. Vocab-size ablation only works when training
#     a Transformer from scratch (not feasible on Kaggle T4 in one session).
#
#   This version's fix:
#     The ablation variable is changed to TRAINING DATA SIZE:
#       - "small"  : 20,000 pairs  (~4% of available data)
#       - "medium" : 100,000 pairs (~20% of available data)
#       - "full"   : all available pairs
#
#     This produces genuinely different fine-tuned models and genuine BLEU
#     score differences. It also answers a scientifically valid and interesting
#     question: "How much cleaned data is required to recover the BLEU
#     degradation caused by raw fine-tuning (Approach 2)?"
#
#   Other bugs fixed:
#     - Removed `import tokenizer` (imported wrong module; was unused)
#     - Changed processing_class= → tokenizer= for broad HF version compat
#     - Added random seed to data subsampling for reproducibility
#
# USAGE (Kaggle):
#   python approach3.py --base-dir /kaggle/working/data \
#                       --results-dir /kaggle/working/results
#
# ESTIMATED RUNTIME: ~4 hours on Kaggle T4
#   - Data pipeline:   ~20 min
#   - Tokenizer train: ~15 min  (kept for methodology; output shown in table)
#   - Fine-tune small: ~25 min
#   - Fine-tune med:   ~65 min
#   - Fine-tune full:  ~80 min
#   - Evaluation:      ~15 min
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import inspect
from transformers import Seq2SeqTrainer



import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.download_and_clean import run_pipeline
from tokenizer.train_spm import run_tokenizer_training




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME   = "Helsinki-NLP/opus-mt-ur-en"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
LR           = 2e-5
EPOCHS       = 3
WARMUP_STEPS = 500
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

# Ablation variable: training data size
# None = use all available data
TRAIN_SIZES = {
    "small":  20_000,
    "medium": 100_000,
    "full":   None,
}


# =============================================================================
# STEP 1 — DATA PIPELINE
# =============================================================================

def step1_data(base_dir: str) -> dict:
    """Download, clean, and split OPUS corpora."""
    print("\n" + "=" * 70)
    print("STEP 1 — DATA PIPELINE")
    print("=" * 70)

    final_dir = os.path.join(base_dir, "final")
    train_tsv = os.path.join(final_dir, "train.tsv")

    if os.path.exists(train_tsv):
        print(f"  Data already exists at {final_dir} — skipping download.")
        print(f"  Delete {final_dir} to force re-download.")
    else:
        run_pipeline(base_dir=base_dir)

    splits = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(final_dir, f"{split}.tsv")
        if os.path.exists(path):
            n = sum(1 for _ in open(path, encoding="utf-8"))
            splits[split] = n
            print(f"  {split}.tsv: {n:,} pairs")
        else:
            print(f"  WARNING: {split}.tsv not found")

    return {
        "train_tsv": os.path.join(final_dir, "train.tsv"),
        "val_tsv":   os.path.join(final_dir, "val.tsv"),
        "test_tsv":  os.path.join(final_dir, "test.tsv"),
        "urdu_txt":  os.path.join(final_dir, "urdu_train_only.txt"),
        "splits":    splits,
    }


# =============================================================================
# STEP 2 — TOKENIZER TRAINING (kept for methodology; output shown in table)
# =============================================================================

def step2_tokenizers(base_dir: str, tokenizer_dir: str) -> dict:
    """
    Train SentencePiece BPE models at 8k, 16k, 32k vocab sizes.

    NOTE: These tokenizers are NOT used in fine-tuning (MarianMT has a fixed
    vocabulary and cannot accept a swapped tokenizer). They are trained here
    to demonstrate the methodology and to show how Urdu text is segmented
    differently at each vocab size — useful analysis for the report.
    """
    print("\n" + "=" * 70)
    print("STEP 2 — TOKENIZER TRAINING (for analysis; not used in training)")
    print("=" * 70)

    vocab_sizes = [8_000, 16_000, 32_000]
    run_tokenizer_training(
        vocab_sizes   = vocab_sizes,
        data_dir      = base_dir,
        tokenizer_dir = tokenizer_dir,
    )

    models = {}
    for vs in vocab_sizes:
        label = f"{vs // 1000}k"
        path  = os.path.join(tokenizer_dir, f"urdu_bpe_{label}.model")
        if os.path.exists(path):
            models[label] = path
            print(f"  urdu_bpe_{label}.model ready")
        else:
            print(f"  WARNING: urdu_bpe_{label}.model not found")

    return models


# =============================================================================
# STEP 3 — TOKENIZATION COMPARISON TABLE
# =============================================================================

def step3_tokenization_table(spm_models: dict) -> None:
    """
    Show how representative Urdu sentences are segmented at each vocab size.
    This table demonstrates the motivation for vocab-size research.
    Screenshot this for the report.
    """
    print("\n" + "=" * 70)
    print("STEP 3 — TOKENIZATION COMPARISON")
    print("  (shows segmentation differences; screenshot for report)")
    print("=" * 70)

    try:
        import sentencepiece as spm
    except ImportError:
        print("  sentencepiece not installed, skipping")
        return

    test_sentences = [
        ("short",     "میں آپ سے بات کرنا چاہتا ہوں"),
        ("formal",    "اقوام متحدہ نے اس فیصلے کی مخالفت کی"),
        ("technical", "کمپیوٹر پروگرامنگ کی بنیادی باتیں سیکھنا"),
        ("complex",   "خوبصورتی ایک ایسی چیز ہے جو آنکھوں کو خوش کرتی ہے"),
    ]

    also_show_marian = True
    marian_tok = None
    if also_show_marian:
        try:
            marian_tok = MarianTokenizer.from_pretrained(MODEL_NAME)
        except Exception:
            marian_tok = None

    print(f"\n{'Type':<12} {'Vocab':<14} {'#Tokens':<10} Pieces (first 12)")
    print("-" * 90)

    for label, sent in test_sentences:
        # MarianMT tokenizer (what is actually used in training)
        if marian_tok is not None:
            pieces = marian_tok.tokenize(sent)
            print(f"{label:<12} {'MarianMT':<14} {len(pieces):<10} "
                  f"{' | '.join(pieces[:12])}")

        # Custom SPM tokenizers (for comparison)
        for vocab_k, model_path in spm_models.items():
            sp = spm.SentencePieceProcessor()
            sp.Load(model_path)
            pieces = sp.EncodeAsPieces(sent)
            print(f"{'':12} {vocab_k + ' SPM':<14} {len(pieces):<10} "
                  f"{' | '.join(pieces[:12])}")
        print()


# =============================================================================
# STEP 4 — FINE-TUNING  (ablation on training data size)
# =============================================================================

def load_tsv(path: str, max_rows: int | None = None, seed: int = SEED) -> Dataset:
    """
    Load a TSV file into a HuggingFace Dataset.

    Args:
        path    : Path to the TSV file (urdu\\tenglish per line).
        max_rows: If set, randomly subsample to this many rows.
        seed    : Random seed for reproducible subsampling.
    """
    urdu, english = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                urdu.append(parts[0].strip())
                english.append(parts[1].strip())

    if max_rows is not None and max_rows < len(urdu):
        rng = random.Random(seed)
        indices = rng.sample(range(len(urdu)), max_rows)
        urdu    = [urdu[i]    for i in indices]
        english = [english[i] for i in indices]
        print(f"  Subsampled to {max_rows:,} pairs (seed={seed})")

    return Dataset.from_dict({"urdu": urdu, "english": english})


def build_tokenize_fn(tok, max_len: int = 128):
    def tokenize_fn(batch):
        model_inputs = tok(
            batch["urdu"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        labels = tok(
            text_target=batch["english"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = [
            [(t if t != tok.pad_token_id else -100) for t in lbl]
            for lbl in labels["input_ids"]
        ]
        return model_inputs
    return tokenize_fn


def build_compute_metrics(tok):
    bleu_m = BLEU(tokenize="13a")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        decoded_preds  = [p.strip() for p in tok.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tok.batch_decode(labels, skip_special_tokens=True)]
        return {"bleu": bleu_m.corpus_score(decoded_preds, [decoded_labels]).score}

    return compute_metrics




def step4_finetune(size_label: str, data: dict, output_dir: str) -> str:
    """
    Fine-tune MarianMT using a subset of the cleaned training data.

    Args:
        size_label: One of "small", "medium", "full" — determines how many
                    training pairs are used (see TRAIN_SIZES constant).
        data      : Dict with keys train_tsv, val_tsv (from step1_data).
        output_dir: Root directory for checkpoints.

    Returns:
        Path to the saved checkpoint directory.
    """
    max_train = TRAIN_SIZES[size_label]

    print(f"\n{'=' * 70}")
    print(f"STEP 4 — FINE-TUNING  (data size: {size_label})")
    if max_train:
        print(f"  Training on {max_train:,} pairs")
    else:
        print("  Training on all available pairs")
    print(f"{'=' * 70}")

    checkpoint_dir = os.path.join(output_dir, f"marianmt_{size_label}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Always use the MarianMT tokenizer — it cannot be swapped on a pre-trained model
    tok   = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("  Loading and tokenizing data...")
    train_ds = load_tsv(data["train_tsv"], max_rows=max_train, seed=SEED)
    val_ds   = load_tsv(data["val_tsv"],   max_rows=None)

    tokenize_fn     = build_tokenize_fn(tok, MAX_LENGTH)
    train_tokenized = train_ds.map(tokenize_fn, batched=True,
                                   remove_columns=train_ds.column_names)
    val_tokenized   = val_ds.map(tokenize_fn, batched=True,
                                 remove_columns=val_ds.column_names)

    print(f"  Train: {len(train_tokenized):,} | Val: {len(val_tokenized):,}")

    args = Seq2SeqTrainingArguments(
        output_dir                  = checkpoint_dir,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        learning_rate               = LR,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        num_train_epochs            = EPOCHS,
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = 0.01,
        predict_with_generate       = True,
        generation_max_length       = MAX_LENGTH,
        load_best_model_at_end      = True,
        metric_for_best_model       = "bleu",
        greater_is_better           = True,
        logging_steps               = 100,
        fp16                        = (DEVICE == "cuda"),
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = SEED,
    )

    collator = DataCollatorForSeq2Seq(tok, model=model,
                                      label_pad_token_id=-100,
                                      pad_to_multiple_of=8)

    import inspect
    _trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    _tok_kwarg = "processing_class" if "processing_class" in _trainer_params else "tokenizer"

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = args,
        train_dataset    = train_tokenized,
        eval_dataset     = val_tokenized,
        **{_tok_kwarg: tok},
        data_collator    = collator,
        compute_metrics  = build_compute_metrics(tok),
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    

    t0 = time.time()
    trainer.train()
    trainer.save_model(checkpoint_dir)
    elapsed = (time.time() - t0) / 60
    print(f"\n  Training complete in {elapsed:.1f} min")
    print(f"  Checkpoint saved to {checkpoint_dir}")

    return checkpoint_dir


# =============================================================================
# STEP 5 — EVALUATION
# =============================================================================

def translate_batch(urdu_sentences, model, tok, batch_size=32):
    model.eval()
    translations = []
    for i in range(0, len(urdu_sentences), batch_size):
        batch  = urdu_sentences[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=4, max_length=MAX_LENGTH)
        translations.extend(tok.batch_decode(out, skip_special_tokens=True))
    return translations


def step5_evaluate(size_label: str, checkpoint_dir: str, test_tsv: str) -> dict:
    """Evaluate a fine-tuned model on the full test set."""
    print(f"\n  Evaluating {size_label}...")

    tok   = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(checkpoint_dir).to(DEVICE)

    test_ur, test_en = [], []
    with open(test_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                test_ur.append(parts[0].strip())
                test_en.append(parts[1].strip())

    hyps = translate_batch(test_ur, model, tok)

    bleu_m = BLEU(tokenize="13a")
    chrf_m = CHRF(word_order=2)
    bleu   = bleu_m.corpus_score(hyps, [test_en])
    chrf   = chrf_m.corpus_score(hyps, [test_en])

    print(f"  {size_label}: BLEU={bleu.score:.2f}  ChrF++={chrf.score:.2f}")

    return {
        "label"  : size_label,
        "n_train": TRAIN_SIZES[size_label],
        "bleu"   : round(bleu.score, 2),
        "chrf"   : round(chrf.score, 2),
        "hyps"   : hyps,
        "refs"   : test_en,
        "sources": test_ur,
    }


# =============================================================================
# STEP 6 — RESULTS TABLE + ERROR ANALYSIS
# =============================================================================

def step6_results(all_results: list, results_dir: str) -> None:
    """Print final results table and save error analysis CSVs."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    bleu_m = BLEU(tokenize="13a")

    rows = [
        {"System": "IndicTrans2 (SOTA)",              "BLEU": 30.76, "ChrF++": 53.00,
         "Train pairs": "—",    "Note": "published upper bound"},
        {"System": "MarianMT zero-shot",               "BLEU": 25.35, "ChrF++": 44.86,
         "Train pairs": "0",    "Note": "approach 2 baseline"},
        {"System": "MarianMT + raw fine-tune (50k)",   "BLEU": 21.40, "ChrF++": 42.55,
         "Train pairs": "50k",  "Note": "approach 2, unfiltered data"},
    ]
    for r in all_results:
        n_label = f"{r['n_train']:,}" if r["n_train"] else "all"
        rows.append({
            "System"     : f"MarianMT + cleaned ({r['label']})",
            "BLEU"       : r["bleu"],
            "ChrF++"     : r["chrf"],
            "Train pairs": n_label,
            "Note"       : "approach 3",
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "final_results.csv"), index=False)
    print(f"\n  Saved to {results_dir}/final_results.csv")

    # Error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS (10 worst translations per variant)")
    print("=" * 70)

    for r in all_results:
        sent_scores = [
            bleu_m.sentence_score(h, [ref]).score
            for h, ref in zip(r["hyps"], r["refs"])
        ]
        worst_idx = sorted(range(len(sent_scores)),
                           key=lambda i: sent_scores[i])[:10]

        print(f"\n--- {r['label']} ({TRAIN_SIZES[r['label']] or 'all'} pairs) ---")
        rows_err = []
        for rank, idx in enumerate(worst_idx):
            print(f"  [{rank+1}] UR : {r['sources'][idx]}")
            print(f"       REF: {r['refs'][idx]}")
            print(f"       OUT: {r['hyps'][idx]}")
            print(f"       sBLEU: {sent_scores[idx]:.2f}\n")
            rows_err.append({
                "urdu_source": r["sources"][idx],
                "reference"  : r["refs"][idx],
                "hypothesis" : r["hyps"][idx],
                "sent_bleu"  : sent_scores[idx],
            })

        err_df   = pd.DataFrame(rows_err)
        err_path = os.path.join(results_dir, f"error_analysis_{r['label']}.csv")
        err_df.to_csv(err_path, index=False, encoding="utf-8-sig")

    # Save JSON summary
    metrics_out = {
        "approach2_baseline"    : {"bleu": 25.35, "chrf": 44.86},
        "approach2_raw_finetune": {"bleu": 21.40, "chrf": 42.55},
    }
    for r in all_results:
        metrics_out[f"approach3_{r['label']}"] = {
            "bleu"   : r["bleu"],
            "chrf"   : r["chrf"],
            "n_train": r["n_train"],
        }
    with open(os.path.join(results_dir, "all_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nAll files saved to {results_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Approach 3: cleaned data + training-size ablation"
    )
    parser.add_argument("--base-dir",        default="/kaggle/working/data")
    parser.add_argument("--results-dir",     default="/kaggle/working/results")
    parser.add_argument("--skip-data",       action="store_true",
                        help="Skip data pipeline if TSVs already exist")
    parser.add_argument("--skip-tokenizer",  action="store_true",
                        help="Skip SPM tokenizer training")
    parser.add_argument("--size",            default="all",
                        choices=["all", "small", "medium", "full"],
                        help="Which data-size variant to run (default: all)")
    args = parser.parse_args()

    tokenizer_dir  = os.path.join(args.base_dir, "tokenizers")
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")

    print(f"\nDevice     : {DEVICE}")
    print(f"Base dir   : {args.base_dir}")
    print(f"Results dir: {args.results_dir}")
    print(f"Ablation   : training data size  {TRAIN_SIZES}")

    # Step 1 — Data
    if args.skip_data:
        final_dir = os.path.join(args.base_dir, "final")
        data = {
            "train_tsv": os.path.join(final_dir, "train.tsv"),
            "val_tsv"  : os.path.join(final_dir, "val.tsv"),
            "test_tsv" : os.path.join(final_dir, "test.tsv"),
            "urdu_txt" : os.path.join(final_dir, "urdu_train_only.txt"),
        }
        print("\nSkipping data pipeline (--skip-data)")
    else:
        data = step1_data(args.base_dir)

    # Step 2 — Tokenizers (analysis only)
    if args.skip_tokenizer:
        spm_models = {}
        print("\nSkipping tokenizer training (--skip-tokenizer)")
    else:
        spm_models = step2_tokenizers(args.base_dir, tokenizer_dir)

    # Step 3 — Tokenization table
    if spm_models:
        step3_tokenization_table(spm_models)

    # Which size variants to run
    run_sizes = list(TRAIN_SIZES.keys()) if args.size == "all" else [args.size]

    # Steps 4 + 5 — Fine-tune and evaluate each variant
    all_results = []
    for size_label in run_sizes:
        checkpoint = step4_finetune(size_label, data, checkpoint_dir)
        result     = step5_evaluate(size_label, checkpoint, data["test_tsv"])
        all_results.append(result)

    # Step 6 — Results table + error analysis
    step6_results(all_results, args.results_dir)

    print("\n✓ Approach 3 complete.")


if __name__ == "__main__":
    main()