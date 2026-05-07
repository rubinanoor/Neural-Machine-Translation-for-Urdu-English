# =============================================================================
# data/download_and_clean.py
# =============================================================================
# Main orchestration script for the Urdu-English NMT data pipeline.
#
# WHAT THIS SCRIPT DOES (in order):
#   1. Downloads 8 OPUS parallel corpora for Urdu-English via opustools
#   2. Runs the multi-stage cleaning pipeline on each corpus independently
#   3. Merges cleaned corpora into a combined training set
#   4. Deduplicates within training and across train/val/test splits
#   5. Shuffles the combined training data
#   6. Creates train / val / test splits and saves them to disk
#   7. Writes a stats.json summary and prints a sanity-check report
#
# DIRECTORY STRUCTURE CREATED:
#   {BASE_DIR}/
#     raw/          ← raw .txt files downloaded from OPUS (not committed to git)
#     cleaned/      ← per-corpus cleaned TSV files
#     final/
#       train.tsv              ← combined shuffled training set
#       val.tsv                ← validation set (held out from TED2020)
#       test.tsv               ← test set (Tanzil)
#       urdu_train_only.txt    ← Urdu-only text for SentencePiece training
#       stats.json             ← corpus and split statistics
#
# USAGE:
#   On Kaggle (after cloning the repo):
#       !python urdu-en-nmt/data/download_and_clean.py
#
#   Locally (for testing on a small subset):
#       python data/download_and_clean.py --base-dir ./local_data --dry-run
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from opustools import OpusRead

# Local modules — all in the same data/ package
from .normalization import normalize_urdu, normalize_english
from .cleaning_filters import (
    contains_urdu_script,
    passes_length_filter,
    passes_ratio_filter,
    passes_content_filter,
    passes_language_filter,
)
from .utils import (
    save_tsv,
    load_tsv,
    deduplicate,
    remove_cross_split_overlap,
    print_split_summary,
    print_samples,
)

# Type alias
SentencePair = Tuple[str, str]

# =============================================================================
# CORPUS CONFIGURATION
# =============================================================================
# Each entry defines one OPUS corpus to download and how to route its output.
#
# Fields:
#   name      : Human-readable label (used for filenames and logging)
#   corpus    : OPUS corpus ID — must match opus.nlpl.eu exactly
#   source    : Source language code (ur = Urdu)
#   target    : Target language code (en = English)
#   max_pairs : Cap on cleaned output (None = keep all)
#   role      : "train" | "train_and_val" | "test_only"
#   quality   : "noisy" | "medium" | "clean" | "high"
#               Controls filter thresholds in cleaning_filters.py

CORPORA: List[Dict] = [
    {
        "name"      : "GNOME",
        "corpus"    : "GNOME",
        "source"    : "ur",
        "target"    : "en",
        "max_pairs" : None,
        "role"      : "train",
        "quality"   : "clean",
    },
    {
        "name"      : "KDE4",
        "corpus"    : "KDE4",
        "source"    : "ur",
        "target"    : "en",
        "max_pairs" : None,
        "role"      : "train",
        "quality"   : "clean",
    },
    {
        "name"      : "Ubuntu",
        "corpus"    : "Ubuntu",
        "source"    : "ur",
        "target"    : "en",
        "max_pairs" : None,
        "role"      : "train",
        "quality"   : "clean",
    },
    {
        "name"      : "QED",
        "corpus"    : "QED",
        "source"    : "ur",
        "target"    : "en",
        "max_pairs" : None,
        "role"      : "train",
        "quality"   : "medium",
    },
    {
        "name"      : "TED2020",
        "corpus"    : "TED2020",
        "source"    : "ur",
        "target"    : "en",
        "max_pairs" : None,
        "role"      : "train",
        "quality"   : "high",
    },
    {
        "name"      : "Tanzil",
        "corpus"    : "Tanzil",
        "source"    : "ur",
        "target"    : "en",
        "max_pairs" : None,
        "role"      : "train",
        "quality"   : "high",
    },
]


# =============================================================================
# DOWNLOAD
# =============================================================================

import urllib.request
import zipfile
import io

import urllib.request
import zipfile
import io

def download_opus_corpus(
    corpus_name: str,
    source_lang: str,
    target_lang: str,
    raw_dir: str,
) -> List[SentencePair]:
    """
    Direct download with corrected URL pathing (alphabetical language codes).
    """
    print(f"  [{corpus_name}] Accessing OPUS via direct download...")
    pairs: List[SentencePair] = []
    
    # OPUS URLs almost always use alphabetical order (en-ur, not ur-en)
    # We use en-ur here because that is how the files are named on the server.
    url = f"https://object.pouta.csc.fi/OPUS-{corpus_name}/v1/moses/en-ur.txt.zip"

    try:
        request = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        )
        
        with urllib.request.urlopen(request) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                file_list = z.namelist()
                
                # Dynamically find files ending in .ur and .en
                # This ensures it works even if the zip internal paths vary
                src_file_name = [f for f in file_list if f.endswith(f".{source_lang}")][0]
                tgt_file_name = [f for f in file_list if f.endswith(f".{target_lang}")][0]

                with z.open(src_file_name) as f_src, z.open(tgt_file_name) as f_tgt:
                    src_lines = f_src.read().decode('utf-8').splitlines()
                    tgt_lines = f_tgt.read().decode('utf-8').splitlines()
                    
                    for s, t in zip(src_lines, tgt_lines):
                        if s.strip() and t.strip():
                            pairs.append((s.strip(), t.strip()))
                
        print(f"  [{corpus_name}] Successfully loaded {len(pairs):,} raw pairs.")
        
    except Exception as exc:
        print(f"  [{corpus_name}] DIRECT DOWNLOAD FAILED: {exc}")
        return []

    return pairs


# =============================================================================
# CLEANING PIPELINE
# =============================================================================

def clean_corpus(
    pairs:        List[SentencePair],
    quality:      str = "medium",
    max_pairs:    Optional[int] = None,
    corpus_name:  str = "Unknown",
) -> List[SentencePair]:
    """
    Runs all cleaning filters on a list of raw sentence pairs and returns
    only the pairs that pass every stage.

    PIPELINE STAGES (in execution order):
      0. Normalize both sides (must happen first — filters see clean text)
      1. Urdu script check      — fastest; rejects non-Arabic-script "Urdu"
      2. Length filter          — word-count bounds per quality tier
      3. Ratio filter           — source/target word-count ratio bounds
      4. Content filter         — URLs, identical pairs, ASCII spam, etc.
      5. Language detection     — langdetect on English side (slowest, done last)

    After passing all filters, if max_pairs is set, we take a random sample
    rather than the first N pairs to avoid systematic sampling bias
    (e.g. CCAligned is sorted by URL similarity score).

    Args:
        pairs       : Raw (urdu, english) tuples from download step.
        quality     : Quality tier for threshold selection.
        max_pairs   : Optional cap on the number of cleaned pairs returned.
        corpus_name : Label for progress bar and logging.

    Returns:
        Cleaned list of (urdu, english) tuples.
    """
    cleaned: List[SentencePair] = []
    rejected = {
        "empty"          : 0,
        "not_urdu_script": 0,
        "length"         : 0,
        "ratio"          : 0,
        "content"        : 0,
        "language"       : 0,
    }

    print(f"\nCleaning {corpus_name} ({len(pairs):,} raw pairs, quality='{quality}')...")

    for urdu_raw, english_raw in tqdm(pairs, desc=f"  Filtering {corpus_name}"):

        # --- Guard: skip empty inputs ---
        if not urdu_raw or not english_raw:
            rejected["empty"] += 1
            continue

        # --- Stage 0: Normalize ---
        urdu    = normalize_urdu(urdu_raw.strip())
        english = normalize_english(english_raw.strip())

        # Re-check for empty after normalization
        # (some strings become empty after stripping diacritics/ZWJ characters)
        if not urdu or not english:
            rejected["empty"] += 1
            continue

        # --- Stage 1: Urdu script check ---
        if not contains_urdu_script(urdu):
            rejected["not_urdu_script"] += 1
            continue

        # --- Stage 2: Length filter ---
        if not passes_length_filter(urdu, english, quality):
            rejected["length"] += 1
            continue

        # --- Stage 3: Ratio filter ---
        if not passes_ratio_filter(urdu, english, quality):
            rejected["ratio"] += 1
            continue

        # --- Stage 4: Content filter ---
        if not passes_content_filter(urdu, english):
            rejected["content"] += 1
            continue

        # --- Stage 5: Language detection (slowest — last) ---
        if not passes_language_filter(urdu, english, quality):
            rejected["language"] += 1
            continue

        cleaned.append((urdu, english))

    # --- Cap at max_pairs via random sampling (not head-truncation) ---
    if max_pairs is not None and len(cleaned) > max_pairs:
        cleaned = random.sample(cleaned, max_pairs)

    # --- Print rejection summary ---
    total_rej     = sum(rejected.values())
    retention_pct = len(cleaned) / max(1, len(pairs)) * 100

    print(f"\n  {corpus_name} Cleaning Summary:")
    print(f"    Raw pairs          : {len(pairs):>10,}")
    print(f"    Cleaned pairs      : {len(cleaned):>10,}  ({retention_pct:.1f}% retention)")
    print(f"    Rejected — empty   : {rejected['empty']:>10,}")
    print(f"    Rejected — script  : {rejected['not_urdu_script']:>10,}")
    print(f"    Rejected — length  : {rejected['length']:>10,}")
    print(f"    Rejected — ratio   : {rejected['ratio']:>10,}")
    print(f"    Rejected — content : {rejected['content']:>10,}")
    print(f"    Rejected — langdet : {rejected['language']:>10,}")

    return cleaned


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(base_dir: str, seed: int = 42) -> None:
    """
    Runs the complete download → clean → split → save pipeline.

    Args:
        base_dir : Root output directory (e.g. "/kaggle/working").
        seed     : Random seed for reproducible shuffling and sampling.
    """
    random.seed(seed)

    # --- Directory setup ---
    raw_dir     = os.path.join(base_dir, "raw")
    cleaned_dir = os.path.join(base_dir, "cleaned")
    final_dir   = os.path.join(base_dir, "final")

    for d in [raw_dir, cleaned_dir, final_dir]:
        os.makedirs(d, exist_ok=True)
        print(f"Directory ready: {d}")

    #One master list for all cleaned data
    all_cleaned_pairs: List[SentencePair] = []
    corpus_stats: Dict = {}


    # -------------------------------------------------------------------------
    # Step 1: Download and clean each corpus
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1 — DOWNLOAD AND CLEAN")
    print("=" * 70)

    for cfg in CORPORA:
            name = cfg["name"]
            raw_pairs = download_opus_corpus(cfg["corpus"], cfg["source"], cfg["target"], raw_dir)

            if not raw_pairs:
                continue

            cleaned = clean_corpus(raw_pairs, cfg["quality"], cfg["max_pairs"], name)

            if cleaned:
                all_cleaned_pairs.extend(cleaned)
                save_tsv(cleaned, os.path.join(cleaned_dir, f"{name}_cleaned.tsv"))

            corpus_stats[name] = {
                "raw": len(raw_pairs),
                "cleaned": len(cleaned),
                "retention_pct": round(len(cleaned) / len(raw_pairs) * 100, 1),
                "role": cfg.get("role", "train"),
                "quality": cfg["quality"]
            }

        


    # -------------------------------------------------------------------------
    # Step 2: Global Shuffle and Percentage Split
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 — GLOBAL SHUFFLE & PERCENTAGE SPLIT (98/1/1)")
    print("=" * 70)

    # Shuffle the entire pool so domains are mixed
    random.shuffle(all_cleaned_pairs)

    total_len = len(all_cleaned_pairs)
    
    # Calculate 1% for Val and 1% for Test
    test_size = max(1000, int(total_len * 0.01))
    val_size  = max(1000, int(total_len * 0.01))

    # Slice the master list
    test_pairs  = all_cleaned_pairs[:test_size]
    val_pairs   = all_cleaned_pairs[test_size : test_size + val_size]
    train_pairs = all_cleaned_pairs[test_size + val_size :]

    # -------------------------------------------------------------------------
    # Step 3: Deduplication (Clean the training set specifically)
    # -------------------------------------------------------------------------
    train_pairs = deduplicate(train_pairs, key="source")
    train_pairs = remove_cross_split_overlap(train_pairs, val_pairs + test_pairs)


# -------------------------------------------------------------------------
    # Step 4: Save final splits
    # -------------------------------------------------------------------------
    save_tsv(train_pairs, os.path.join(final_dir, "train.tsv"))
    save_tsv(val_pairs,   os.path.join(final_dir, "val.tsv"))
    save_tsv(test_pairs,  os.path.join(final_dir, "test.tsv"))

    # Save Urdu-only text for tokenizer
    with open(os.path.join(final_dir, "urdu_train_only.txt"), "w", encoding="utf-8") as f:
        for urdu, _ in train_pairs:
            f.write(urdu + "\n")

    # -------------------------------------------------------------------------
    # Step 5: Statistics report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5 — STATISTICS REPORT")
    print("=" * 70)

    # Per-corpus table
    print(f"\n{'Corpus':<20} {'Raw':>10} {'Cleaned':>10} {'Retention':>10} {'Role':<20}")
    print("─" * 75)
    for cname, stats in corpus_stats.items():
        if stats["raw"] > 0:
            # Use .get() to provide a fallback value so it never crashes again
            role_label = stats.get('role', 'combined_pool') 
            retention = stats.get('retention_pct', 0)
            
            print(f"{cname:<20} {stats['raw']:>10,} {stats['cleaned']:>10,} "
                  f"{retention:>9.1f}%  {role_label:<20}")

    # Split summary with average lengths
    print_split_summary(train_pairs, val_pairs, test_pairs)

    # Save stats.json
    final_stats = {
        "splits"    : {
            "train"     : len(train_pairs),
            "validation": len(val_pairs),
            "test"      : len(test_pairs),
            "total"     : len(train_pairs) + len(val_pairs) + len(test_pairs),
        },
        "per_corpus": corpus_stats,
    }
    stats_path = os.path.join(final_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Statistics saved → {stats_path}")

    # -------------------------------------------------------------------------
    # Step 6: Sanity check — print random samples from each split
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6 — SANITY CHECK (RANDOM SAMPLES)")
    print("=" * 70)
    print_samples(train_pairs, "TRAINING SET")
    print_samples(val_pairs,   "VALIDATION SET")
    print_samples(test_pairs,  "TEST SET")

    # -------------------------------------------------------------------------
    # Step 7: List output files
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    for directory in [cleaned_dir, final_dir]:
        print(f"\n{directory}/")
        for fname in sorted(os.listdir(directory)):
            fpath    = os.path.join(directory, fname)
            size_mb  = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fname:<40} {size_mb:>8.2f} MB")

    print("\n✓ Pipeline complete. Files are ready for model training.")

    # -------------------------------------------------------------------------
    # Usage reminder
    # -------------------------------------------------------------------------
    print("""
================================================================================
NEXT STEPS
================================================================================

1. Load data in your training notebook:

    from datasets import Dataset
    import pandas as pd

    def tsv_to_dataset(path):
        df = pd.read_csv(path, sep='\\t', names=['urdu', 'english'])
        return Dataset.from_pandas(df)

    train_ds = tsv_to_dataset(f"{final_dir}/train.tsv")
    val_ds   = tsv_to_dataset(f"{final_dir}/val.tsv")
    test_ds  = tsv_to_dataset(f"{final_dir}/test.tsv")

2. Train SentencePiece BPE tokenizer (vocabulary ablation):

    import sentencepiece as spm

    for vocab_size in [8000, 16000, 32000]:
        spm.SentencePieceTrainer.train(
            input              = f"{final_dir}/urdu_train_only.txt",
            model_prefix       = f"spm_bpe_{vocab_size}",
            vocab_size         = vocab_size,
            character_coverage = 1.0,   # Critical for Urdu script
            model_type         = "bpe",
        )

3. Files reference:
    {final_dir}/train.tsv           ← Training set
    {final_dir}/val.tsv             ← Validation set
    {final_dir}/test.tsv            ← Test set — DO NOT touch until final eval
    {final_dir}/urdu_train_only.txt ← Urdu text for tokenizer training
    {final_dir}/stats.json          ← Corpus statistics for your report
    {cleaned_dir}/*.tsv             ← Per-corpus cleaned files (for debugging)
================================================================================
""".format(final_dir=final_dir, cleaned_dir=cleaned_dir, vocab_size=8000))


# =============================================================================
# ENTRY POINT
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, clean, and split Urdu-English NMT corpora from OPUS."
    )
    parser.add_argument(
        "--base-dir",
        default="/kaggle/working",
        help="Root output directory (default: /kaggle/working)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(base_dir=args.base_dir, seed=args.seed)