# =============================================================================
# model/evaluate.py
# =============================================================================
# Baseline evaluation: load Helsinki-NLP/opus-mt-ur-en (pre-trained MarianMT),
# run beam-search translation on 2,000 test sentences, and report BLEU + ChrF++.
#
# THIS IS THE KEY SCRIPT FOR THE MID-REPORT.
# Running this script produces the concrete experimental result required:
#   "Baseline MarianMT BLEU on 2k test sentences."
#
# WHAT MARIANMT IS:
#   MarianMT (Junczys-Dowmunt et al., 2018) is a transformer-based sequence-
#   to-sequence model trained with the Marian C++ framework. HuggingFace
#   hosts hundreds of pre-trained checkpoints. opus-mt-ur-en was trained by
#   the Helsinki NLP group directly on OPUS Urdu-English parallel data —
#   the same corpora we cleaned — making it the most appropriate baseline.
#
# EVALUATION METRICS:
#   BLEU  (Papineni et al., 2002) — dominant NMT metric; n-gram precision
#         weighted by brevity penalty. Reported as corpus-level BLEU.
#   ChrF++ (Popović, 2015/2017)   — character n-gram F-score, more sensitive
#         to morphological correctness. Useful for Urdu's rich morphology.
#
# HOW TO RUN (Kaggle GPU cell):
#   !python model/evaluate.py --variant baseline --samples 2000
#
# HOW TO RUN (local CPU, smaller sample for debugging):
#   python model/evaluate.py --variant baseline --samples 100
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

# Internal imports
# Note: when running from Kaggle, add the project root to sys.path first:
#   import sys; sys.path.insert(0, "/kaggle/working/urdu-en-nmt")
from model.config import BaselineConfig, get_config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_pairs(
    tsv_path: str,
    n_samples: int,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Loads (urdu_sources, english_references) from a TSV file and
    returns a random sample of `n_samples` pairs.

    WHY RANDOM SAMPLING:
      Our test split may have up to 10k sentences, but running all of them
      on a Kaggle T4 takes ~20 minutes. 2,000 samples produce a statistically
      stable BLEU estimate (bootstrapped 95 % CI ≈ ±0.5 BLEU at N=2000).

    Args:
        tsv_path : Path to the test.tsv file (format: urdu\tenglish per line).
        n_samples: How many pairs to evaluate on.
        seed     : Controls which 2k pairs are selected (for reproducibility).

    Returns:
        sources    : List of Urdu source strings.
        references : List of English reference strings (ground truth).
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"Test TSV not found: {tsv_path}\n"
            "Have you run data/download_and_clean.py to produce the splits?"
        )

    pairs: List[Tuple[str, str]] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t", 1)   # split on first tab only
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
            # Skip malformed lines silently — robust to minor file corruption

    print(f"  Loaded {len(pairs):,} total test pairs from {tsv_path}")

    if len(pairs) < n_samples:
        print(
            f"  WARNING: only {len(pairs)} pairs available; "
            f"evaluating on all of them (requested {n_samples})."
        )
        n_samples = len(pairs)

    # Use a seeded RNG so the same 2k pairs are always selected.
    # This is critical: if you re-run the evaluation and accidentally select
    # different sentences, BLEU scores become incomparable across runs.
    rng = random.Random(seed)
    sampled = rng.sample(pairs, n_samples)

    sources    = [ur for ur, _  in sampled]
    references = [en for _,  en in sampled]

    print(f"  Sampled {len(sources):,} pairs (seed={seed})")
    return sources, references


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    model_name_or_path: str,
    device: str,
) -> Tuple[MarianMTModel, MarianTokenizer]:
    """
    Downloads (or loads from cache) the MarianMT model and its tokenizer.

    CACHING:
      HuggingFace caches models to ~/.cache/huggingface/hub/ by default.
      On Kaggle, the first run downloads ~300 MB; subsequent runs are instant.
      To use a custom cache dir: set HF_HOME env variable before importing.

    Args:
        model_name_or_path: HuggingFace hub ID or local checkpoint path.
        device            : "cuda", "cpu", or "auto".

    Returns:
        (model, tokenizer) — both moved to the correct device.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Loading tokenizer: {model_name_or_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)

    print(f"  Loading model: {model_name_or_path}")
    model = MarianMTModel.from_pretrained(model_name_or_path)
    model = model.to(device)

    # Always set eval mode before inference.
    # This disables dropout layers — crucial for deterministic BLEU scores.
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded on {device} ({n_params / 1e6:.1f}M parameters)")

    return model, tokenizer


# =============================================================================
# BATCH TRANSLATION
# =============================================================================

def translate_batch(
    sources: List[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    device: str,
    batch_size: int = 32,
    num_beams: int = 4,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> List[str]:
    """
    Translates a list of Urdu source sentences to English using beam search.

    WHY BATCHED INFERENCE:
      Sending 2,000 sentences one-by-one would be 50–100× slower than batching.
      The batch size controls GPU memory usage — 32 is safe for 16 GB GPUs.
      If you get CUDA OOM, reduce to 16.

    WHY BEAM SEARCH (num_beams=4):
      Greedy decoding is fast but 2–3 BLEU points worse than beam-4.
      The NMT community standard for published baselines is beam-4 with
      no length penalty. We match this convention so our results are comparable
      to numbers in related work.

    MARIANMT PREFIX REQUIREMENT:
      MarianMT uses a >>lang<< prefix token to specify the target language.
      For opus-mt-ur-en this is usually not required (single-target model),
      but we add no prefix here — the model's own config handles routing.

    Args:
        sources           : List of Urdu source strings.
        model             : Loaded MarianMTModel (already on device).
        tokenizer         : Corresponding MarianTokenizer.
        device            : "cuda" or "cpu".
        batch_size        : Sentences per forward pass.
        num_beams         : Beam search width.
        max_source_length : Truncate Urdu input to this many subword tokens.
        max_target_length : Maximum output length in subword tokens.

    Returns:
        List of translated English strings (same order as sources).
    """
    translations: List[str] = []
    total_batches = (len(sources) + batch_size - 1) // batch_size

    print(f"\n  Translating {len(sources):,} sentences "
          f"(batch_size={batch_size}, beam_width={num_beams})...")
    t0 = time.time()

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, len(sources))
        batch       = sources[batch_start:batch_end]

        # -----------------------------------------------------------------
        # Tokenize the Urdu batch
        # -----------------------------------------------------------------
        # padding=True    : pad shorter sentences to the longest in this batch
        # truncation=True : trim sentences longer than max_source_length
        # return_tensors  : return PyTorch tensors, not lists
        # -----------------------------------------------------------------
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        # Move input tensors to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # -----------------------------------------------------------------
        # Generate translations
        # -----------------------------------------------------------------
        # torch.no_grad() prevents PyTorch from tracking gradients during
        # inference — halves memory usage and speeds up forward passes.
        # -----------------------------------------------------------------
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_target_length,
                early_stopping=True,    # Stop beam search when all beams hit EOS
            )

        # -----------------------------------------------------------------
        # Decode output token IDs back to text
        # -----------------------------------------------------------------
        # skip_special_tokens=True removes <pad>, </s>, etc. from output.
        batch_translations = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        translations.extend(batch_translations)

        # Progress reporting every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            sents_done = batch_end
            sps = sents_done / elapsed   # sentences per second
            remaining = (len(sources) - sents_done) / max(sps, 1)
            print(
                f"    [{batch_idx + 1}/{total_batches}] "
                f"{sents_done}/{len(sources)} sentences | "
                f"{sps:.1f} sent/s | ETA {remaining:.0f}s"
            )

    elapsed_total = time.time() - t0
    print(f"  Translation complete in {elapsed_total:.1f}s "
          f"({len(sources) / elapsed_total:.1f} sent/s)")
    return translations


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(
    hypotheses: List[str],
    references: List[str],
) -> dict:
    """
    Computes corpus-level BLEU and ChrF++ scores.

    BLEU SCORE INTERPRETATION (rough guide for Urdu→English):
        < 10  : Almost unusable translation quality
        10–20 : Understandable, many errors
        20–30 : Good machine translation
        30–40 : High quality (often hard to distinguish from human at glance)
        > 40  : Near-human (rare without massive data)

    Expected range for opus-mt-ur-en zero-shot baseline: 20–35 BLEU,
    depending on test set domain (TED2020 ≈ higher, Tanzil ≈ lower).

    WHY CHRF++:
      BLEU counts exact n-gram matches. ChrF++ operates at character level,
      making it more robust for morphologically rich languages. It often
      correlates better with human judgments for Urdu than BLEU does.

    Args:
        hypotheses: List of model-generated English translations.
        references : List of reference (human) English translations.

    Returns:
        Dictionary with "bleu", "chrf", "num_sentences", and per-n-gram precisions.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch: {len(hypotheses)} hypotheses vs "
            f"{len(references)} references"
        )

    # -----------------------------------------------------------------
    # sacrebleu expects references wrapped in a list-of-lists:
    # [[ref1, ref2, ...]] — outer list for multiple references, inner for sentences.
    # We have single references, so it becomes [[ref1, ref2, ...]].
    # -----------------------------------------------------------------
    bleu_result = sacrebleu.corpus_bleu(
        hypotheses,
        [references],      # list of reference lists (single reference here)
        tokenize="intl",   # International tokenization — handles Unicode properly
    )

    chrf_result = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2,  # ChrF++ uses word n-grams (order=2); ChrF uses order=0
    )

    metrics = {
        "bleu":          round(bleu_result.score, 2),
        "chrf_plus_plus": round(chrf_result.score, 2),
        "bleu_1gram":    round(bleu_result.precisions[0], 2),
        "bleu_2gram":    round(bleu_result.precisions[1], 2),
        "bleu_3gram":    round(bleu_result.precisions[2], 2),
        "bleu_4gram":    round(bleu_result.precisions[3], 2),
        "bp":            round(bleu_result.bp, 4),     # Brevity penalty
        "num_sentences": len(hypotheses),
    }

    return metrics


# =============================================================================
# RESULT PERSISTENCE
# =============================================================================

def save_results(
    hypotheses: List[str],
    references: List[str],
    sources:    List[str],
    metrics:    dict,
    cfg:        BaselineConfig,
) -> None:
    """
    Saves translations, references, and metrics to the results/ directory.

    FILES PRODUCED:
      baseline_predictions.txt — one model translation per line
      baseline_references.txt  — one reference translation per line
      baseline_metrics.json    — BLEU, ChrF++, and metadata in JSON
      baseline_samples.txt     — 20 side-by-side examples for the report

    WHY SAVE PREDICTIONS:
      BLEU is deterministic given the same predictions file. Saving translations
      allows you to re-compute metrics without re-running the model, and to
      run additional analysis (e.g., error analysis for the report).

    Args:
        hypotheses: Model-generated English strings.
        references : Reference English strings.
        sources    : Original Urdu source strings.
        metrics    : Dictionary from compute_metrics().
        cfg        : BaselineConfig (provides output paths).
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    # --- Save translations ---
    with open(cfg.predictions_file, "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses))
    print(f"  Saved predictions → {cfg.predictions_file}")

    # --- Save references ---
    with open(cfg.references_file, "w", encoding="utf-8") as f:
        f.write("\n".join(references))
    print(f"  Saved references  → {cfg.references_file}")

    # --- Save metrics as JSON ---
    # Add experiment metadata alongside the scores
    full_metrics = {
        "experiment": cfg.experiment_name,
        "model":      cfg.model_name_or_path,
        "num_beams":  cfg.num_beams,
        "eval_samples": cfg.eval_samples,
        "seed":       cfg.seed,
        **metrics,    # BLEU, ChrF++, etc.
    }
    with open(cfg.metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)
    print(f"  Saved metrics     → {cfg.metrics_file}")

    # --- Save human-readable side-by-side sample ---
    samples_file = os.path.join(cfg.results_dir, "baseline_samples.txt")
    rng = random.Random(cfg.seed)
    sample_indices = rng.sample(range(len(sources)), min(20, len(sources)))

    with open(samples_file, "w", encoding="utf-8") as f:
        f.write(f"Baseline MarianMT — 20 Random Translation Samples\n")
        f.write(f"Model : {cfg.model_name_or_path}\n")
        f.write(f"BLEU  : {metrics['bleu']:.2f}\n")
        f.write(f"ChrF++: {metrics['chrf_plus_plus']:.2f}\n")
        f.write("=" * 72 + "\n\n")
        for i, idx in enumerate(sample_indices, 1):
            f.write(f"[{i:02d}] Urdu source : {sources[idx]}\n")
            f.write(f"      Model output: {hypotheses[idx]}\n")
            f.write(f"      Reference   : {references[idx]}\n\n")

    print(f"  Saved samples     → {samples_file}")


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_results_table(metrics: dict, cfg: BaselineConfig) -> None:
    """
    Pretty-prints the evaluation results for quick inspection in a terminal
    or Kaggle notebook output cell.

    Args:
        metrics : Dictionary from compute_metrics().
        cfg     : Config used for the experiment.
    """
    width = 52
    print("\n" + "=" * width)
    print(f"  BASELINE EVALUATION RESULTS")
    print("=" * width)
    print(f"  Model          : {cfg.model_name_or_path}")
    print(f"  Test sentences : {metrics['num_sentences']:,}")
    print(f"  Beam width     : {cfg.num_beams}")
    print("-" * width)
    print(f"  BLEU           : {metrics['bleu']:.2f}")
    print(f"  ChrF++         : {metrics['chrf_plus_plus']:.2f}")
    print("-" * width)
    print(f"  BLEU 1-gram    : {metrics['bleu_1gram']:.2f}")
    print(f"  BLEU 2-gram    : {metrics['bleu_2gram']:.2f}")
    print(f"  BLEU 3-gram    : {metrics['bleu_3gram']:.2f}")
    print(f"  BLEU 4-gram    : {metrics['bleu_4gram']:.2f}")
    print(f"  Brevity Penalty: {metrics['bp']:.4f}")
    print("=" * width)
    print(f"\n  Results saved to: {cfg.results_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def run_baseline_evaluation(cfg: BaselineConfig) -> dict:
    """
    Orchestrates the full baseline evaluation pipeline.

    PIPELINE:
      1. Load 2k test pairs from TSV
      2. Download/cache MarianMT model
      3. Translate Urdu → English with beam-4
      4. Compute BLEU + ChrF++
      5. Save results to results/
      6. Print summary table

    Args:
        cfg: BaselineConfig instance controlling all settings.

    Returns:
        Metrics dictionary {"bleu": float, "chrf_plus_plus": float, ...}
    """
    print("\n" + "=" * 60)
    print("  Urdu→English NMT — Baseline MarianMT Evaluation")
    print("  Mid-Report Experiment")
    print("=" * 60)

    # Step 1: Load test data
    print("\n[1/5] Loading test data...")
    sources, references = load_test_pairs(
        tsv_path  = cfg.test_tsv,
        n_samples = cfg.eval_samples,
        seed      = cfg.seed,
    )

    # Step 2: Load model
    print("\n[2/5] Loading MarianMT model...")
    device = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else \
             cfg.device if cfg.device != "auto" else "cpu"
    model, tokenizer = load_model(cfg.model_name_or_path, device)

    # Step 3: Translate
    print("\n[3/5] Running inference...")
    hypotheses = translate_batch(
        sources           = sources,
        model             = model,
        tokenizer         = tokenizer,
        device            = device,
        batch_size        = cfg.batch_size,
        num_beams         = cfg.num_beams,
        max_source_length = cfg.max_source_length,
        max_target_length = cfg.max_target_length,
    )

    # Step 4: Compute metrics
    print("\n[4/5] Computing BLEU and ChrF++ metrics...")
    metrics = compute_metrics(hypotheses, references)

    # Step 5: Save results
    print("\n[5/5] Saving results...")
    save_results(hypotheses, references, sources, metrics, cfg)

    # Print summary
    print_results_table(metrics, cfg)

    return metrics


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline MarianMT BLEU evaluation on Urdu-English test set."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "vocab_8k", "vocab_16k", "vocab_32k"],
        help="Config variant to use. 'baseline' = zero-shot MarianMT (for mid-report).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2_000,
        help="Number of test sentences to evaluate on (default: 2000).",
    )
    parser.add_argument(
        "--test_tsv",
        type=str,
        default=None,
        help="Override path to test.tsv (optional; defaults to config path).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size. Reduce if CUDA OOM (default: 32).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam search width (default: 4).",
    )
    args = parser.parse_args()

    # Load the config and apply CLI overrides
    cfg = get_config(args.variant)
    cfg.eval_samples = args.samples
    cfg.batch_size   = args.batch_size
    cfg.num_beams    = args.num_beams
    if args.test_tsv:
        cfg.test_tsv = args.test_tsv

    # Run evaluation
    metrics = run_baseline_evaluation(cfg)

    print(f"\nFinal BLEU: {metrics['bleu']:.2f}")
    print(f"Final ChrF++: {metrics['chrf_plus_plus']:.2f}")