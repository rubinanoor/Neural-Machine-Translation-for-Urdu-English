# =============================================================================
# data/utils.py
# =============================================================================
# Shared I/O helpers and deduplication for the NMT data pipeline.
#
# All functions here are pure utilities with no dependency on corpus config
# or cleaning logic — they can be imported by any other module safely.
# =============================================================================

from __future__ import annotations

import random
from typing import List, Tuple

# Type alias for a sentence pair
SentencePair = Tuple[str, str]


# =============================================================================
# TSV I/O
# =============================================================================

def save_tsv(pairs: List[SentencePair], filepath: str) -> None:
    """
    Saves a list of (urdu, english) pairs to a UTF-8 TSV file.

    FORMAT: one pair per line, fields separated by a TAB character.
    Example line:  یہ ایک مثال ہے\tThis is an example

    WHY TSV:
    Urdu text frequently contains commas, so CSV is unreliable. Tabs are
    essentially never used inside sentence text, making them safe delimiters.
    TSV is also natively supported by HuggingFace `datasets` and pandas.

    Args:
        pairs    : List of (urdu_str, english_str) tuples.
        filepath : Destination file path (conventionally *.tsv).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for urdu, english in pairs:
            f.write(f"{urdu}\t{english}\n")

    print(f"  Saved {len(pairs):,} pairs → {filepath}")


def load_tsv(filepath: str) -> List[SentencePair]:
    """
    Loads a TSV file back into a list of (urdu, english) tuples.
    Inverse of save_tsv().

    EDGE CASE HANDLING:
    If a text field itself contains a tab character (rare but possible in
    web-scraped data), the first field is treated as Urdu and everything
    after the first tab is joined as English.

    Args:
        filepath: Path to a *.tsv file.

    Returns:
        List of (urdu_str, english_str) tuples.
    """
    pairs: List[SentencePair] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n')
            parts = line.split('\t')

            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
            elif len(parts) > 2:
                # Text contains a literal tab — preserve it in the English side
                pairs.append((parts[0], '\t'.join(parts[1:])))
            else:
                print(f"  WARNING: Skipping malformed line {line_num} in {filepath}")

    return pairs


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate(
    pairs: List[SentencePair],
    key: str = "source",
) -> List[SentencePair]:
    """
    Removes duplicate sentence pairs from a list.

    WHY DEDUPLICATION IS CRITICAL:
    Several OPUS corpora overlap (GNOME/KDE4 share software strings;
    CCAligned/OpenSubtitles share web content). Duplicates cause:
      • Data leakage if the same pair ends up in train AND test
      • Model bias toward frequently repeated pairs
      • Wasted GPU compute re-training on identical examples

    KEY OPTIONS:
      "source"  — deduplicate on the Urdu side only (recommended).
                  Each unique Urdu sentence appears at most once.
                  Multiple English translations of the same Urdu are merged.
      "target"  — deduplicate on the English side only.
      "both"    — deduplicate on the exact (urdu, english) pair.
                  Most conservative; keeps alternative translations of same source.

    WHY "source" IS DEFAULT:
    We want each unique Urdu sentence to have exactly one English target
    during training. Multiple translations of the same source give the model
    inconsistent training signal for the same input.

    Args:
        pairs: List of (urdu_str, english_str) tuples.
        key  : "source", "target", or "both".

    Returns:
        Deduplicated list, preserving the original order of first occurrences.
    """
    print(f"\nDeduplicating {len(pairs):,} pairs (key='{key}')...")

    seen: set = set()
    deduplicated: List[SentencePair] = []

    for urdu, english in pairs:
        if key == "source":
            lookup = urdu.strip().lower()
        elif key == "target":
            lookup = english.strip().lower()
        else:   # "both"
            lookup = (urdu.strip().lower(), english.strip().lower())

        if lookup not in seen:
            seen.add(lookup)
            deduplicated.append((urdu, english))

    removed = len(pairs) - len(deduplicated)
    pct     = removed / max(1, len(pairs)) * 100
    print(f"  Removed {removed:,} duplicates ({pct:.1f}%)")
    print(f"  Remaining: {len(deduplicated):,} unique pairs")

    return deduplicated


def remove_cross_split_overlap(
    train_pairs: List[SentencePair],
    reference_pairs: List[SentencePair],
) -> List[SentencePair]:
    """
    Removes from `train_pairs` any pair whose Urdu source sentence also
    appears in `reference_pairs` (typically val + test combined).

    WHY THIS IS SEPARATE FROM deduplicate():
    deduplicate() works within a single list. This function works across
    two lists and is specifically designed to prevent data leakage between
    the training set and held-out evaluation sets.

    If a Tanzil sentence appears in both training and test, the model may
    have memorized it — inflating BLEU scores artificially.

    Args:
        train_pairs     : The training set to clean.
        reference_pairs : Val + test pairs to check against.

    Returns:
        Filtered training pairs with no overlap on the Urdu source side.
    """
    print(f"\nRemoving val/test sentences from training set...")

    reference_urdu = {urdu.strip().lower() for urdu, _ in reference_pairs}

    original_size    = len(train_pairs)
    filtered         = [
        (ur, en) for ur, en in train_pairs
        if ur.strip().lower() not in reference_urdu
    ]
    removed          = original_size - len(filtered)

    print(f"  Removed {removed:,} training pairs found in val/test sets")
    print(f"  Final training size: {len(filtered):,}")

    return filtered


# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def avg_word_count(pairs: List[SentencePair], side: int = 0) -> float:
    """
    Returns the mean word count for one side of a list of sentence pairs.

    Args:
        pairs: List of (urdu_str, english_str) tuples.
        side : 0 for Urdu, 1 for English.

    Returns:
        Mean word count as a float, or 0.0 if the list is empty.
    """
    if not pairs:
        return 0.0
    return sum(len(p[side].split()) for p in pairs) / len(pairs)


def print_split_summary(
    train_pairs: List[SentencePair],
    val_pairs:   List[SentencePair],
    test_pairs:  List[SentencePair],
) -> None:
    """
    Prints a human-readable summary table of all three splits.

    Args:
        train_pairs: Training sentence pairs.
        val_pairs  : Validation sentence pairs.
        test_pairs : Test sentence pairs.
    """
    total = len(train_pairs) + len(val_pairs) + len(test_pairs)

    print(f"\n{'Split':<15} {'Pairs':>12}  {'Avg UR len':>10}  {'Avg EN len':>10}")
    print("─" * 55)
    for name, split in [("Train", train_pairs), ("Validation", val_pairs), ("Test", test_pairs)]:
        print(
            f"{name:<15} {len(split):>12,}  "
            f"{avg_word_count(split, 0):>10.1f}  "
            f"{avg_word_count(split, 1):>10.1f}"
        )
    print("─" * 55)
    print(f"{'TOTAL':<15} {total:>12,}")


def print_samples(
    pairs: List[SentencePair],
    split_name: str,
    n: int = 5,
    seed: int = 42,
) -> None:
    """
    Prints `n` random sample pairs from a split for visual inspection.

    Always visually inspect samples after cleaning. If you see garbage here,
    something went wrong in the pipeline.

    Args:
        pairs      : List of (urdu_str, english_str) tuples.
        split_name : Label printed in the header (e.g. "TRAINING SET").
        n          : Number of samples to print.
        seed       : Random seed for reproducible sampling.
    """
    rng     = random.Random(seed)
    samples = rng.sample(pairs, min(n, len(pairs)))

    print(f"\n--- {split_name} (showing {len(samples)} random samples) ---")
    for i, (ur, en) in enumerate(samples, 1):
        print(f"  [{i}] Urdu   : {ur}")
        print(f"       English: {en}")
        print()