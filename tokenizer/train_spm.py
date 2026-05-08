# =============================================================================
# tokenizer/train_spm.py
# =============================================================================
# Trains three SentencePiece BPE tokenizer models on the Urdu training corpus,
# one for each vocabulary size: 8,000 / 16,000 / 32,000 subwords.
#
# WHAT SENTENCEPIECE BPE IS:
#   Byte-Pair Encoding (Sennrich et al., 2016) is the dominant subword
#   tokenization algorithm for NMT. It starts from individual characters and
#   iteratively merges the most frequent adjacent pair into a new token.
#   The result is a vocabulary of "subword units" that handle rare and OOV
#   words gracefully by decomposing them into known sub-parts.
#
#   SentencePiece (Kudo & Richardson, 2018) implements BPE (and other
#   algorithms) in a language-agnostic way, treating the input as a raw
#   Unicode stream — ideal for Urdu which mixes Arabic script with numbers
#   and occasional Latin characters.
#
# WHY TRAIN ON URDU ONLY:
#   MarianMT already has an English-optimized tokenizer baked in.
#   Our ablation question is: how does Urdu BPE vocabulary size affect
#   translation quality? We isolate this by only varying the Urdu tokenizer
#   and keeping the English-side tokenizer constant.
#
# FOR THE MID-REPORT:
#   You do NOT need to run this script. It is needed only for Phase 2.
#   Include it in your submission to demonstrate the complete methodology.
#
# HOW TO RUN (Kaggle, after data pipeline is complete):
#   !python tokenizer/train_spm.py --all
#   or a single size:
#   !python tokenizer/train_spm.py --vocab_size 16000
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
import time

import sentencepiece as spm   # pip install sentencepiece

# Internal imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from model.config import DATA_DIR, TOKENIZER_DIR


# =============================================================================
# CORPUS EXTRACTION
# =============================================================================

def extract_urdu_corpus(
    train_tsv: str,
    output_txt: str,
) -> int:
    """
    Extracts the Urdu column from train.tsv and writes it to a plain text
    file for SentencePiece training.

    SentencePiece's trainer expects a plain text file with one sentence per
    line. Our train.tsv has tab-separated pairs, so we extract just the Urdu
    column here.

    Args:
        train_tsv  : Path to the training TSV file.
        output_txt : Destination path for the Urdu corpus text file.

    Returns:
        Number of sentences written.
    """
    if not os.path.exists(train_tsv):
        raise FileNotFoundError(
            f"Training TSV not found: {train_tsv}\n"
            "Run data/download_and_clean.py first."
        )

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    count = 0
    with open(train_tsv, "r", encoding="utf-8") as fin, \
         open(output_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                fout.write(parts[0].strip() + "\n")
                count += 1

    print(f"  Extracted {count:,} Urdu sentences → {output_txt}")
    return count


# =============================================================================
# SPM TRAINING
# =============================================================================

def train_spm_model(
    input_txt:    str,
    model_prefix: str,
    vocab_size:   int,
    character_coverage: float = 0.9995,
) -> None:
    """
    Trains a SentencePiece BPE model on the Urdu corpus.

    KEY PARAMETERS EXPLAINED:

    vocab_size (8000/16000/32000):
      The number of distinct subword tokens in the final vocabulary.
      Smaller vocab → more aggressive segmentation (more characters/pieces per word).
      Larger vocab → more whole-word or morpheme-level tokens.
      Trade-off: larger vocab needs more training data to learn good merges.

    character_coverage (0.9995):
      Fraction of Unicode characters that must be included in the vocabulary.
      Urdu has ~100 base characters + diacritics + presentation forms.
      0.9995 ensures essentially all legitimate Urdu characters are covered.
      (Default 0.9995 is recommended for non-Latin scripts.)

    model_type="bpe":
      Byte-Pair Encoding. The alternative is "unigram" (the other common
      SentencePiece algorithm). BPE is more widely used in NMT and is the
      algorithm used in the original Transformer paper.

    pad_id / unk_id / bos_id / eos_id:
      Reserve special token IDs at fixed positions so that the NMT model can
      rely on them being consistent across all three vocab sizes. This is
      important for the ablation: when we swap tokenizers, we don't want
      the special tokens to shift positions.

    split_by_unicode_script=True:
      Prevents BPE from creating cross-script merges (e.g., merging a Urdu
      character with a Latin digit). This keeps Urdu and Latin tokens separate,
      which is linguistically sensible.

    Args:
        input_txt          : Path to the plain-text Urdu corpus.
        model_prefix       : Output prefix. Produces {prefix}.model + {prefix}.vocab
        vocab_size         : Number of BPE subword tokens.
        character_coverage : Fraction of characters to cover (default: 0.9995).
    """
    print(f"\n  Training SPM: vocab_size={vocab_size:,} → {model_prefix}")
    t0 = time.time()

    spm.SentencePieceTrainer.train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",              # Byte-Pair Encoding
        character_coverage=character_coverage,

        # Special tokens — fixed IDs for compatibility across vocab sizes
        pad_id=0,                      # <pad> for sequence padding
        unk_id=1,                      # <unk> for out-of-vocabulary characters
        bos_id=2,                      # <s>  beginning of sequence
        eos_id=3,                      # </s> end of sequence

        # Urdu-specific settings
        split_by_unicode_script=True,  # Don't merge across script boundaries
        byte_fallback=True,            # Fall back to raw bytes for OOV chars

        # Normalization
        normalization_rule_name="nmt_nfkc",  # Standard NMT Unicode normalization

        # Training efficiency
        shuffle_input_sentence=True,   # Shuffle before training for better merges
        num_threads=4,                 # Parallel BPE merge counting
        input_sentence_size=2_000_000, # Max sentences to use (cap for speed)
        train_extremely_large_corpus=False,

        # Logging
        minloglevel=1,  # 0=INFO, 1=WARNING, 2=ERROR
    )

    elapsed = time.time() - t0
    model_file = model_prefix + ".model"
    vocab_file  = model_prefix + ".vocab"
    model_size_mb = os.path.getsize(model_file) / 1e6

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Model file  : {model_file} ({model_size_mb:.2f} MB)")
    print(f"  Vocab file  : {vocab_file}")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_spm_model(model_path: str, test_sentences: list[str]) -> None:
    """
    Loads the trained SPM model and tokenizes a few sample sentences.
    Prints token counts to help identify over-segmentation.

    WHAT TO LOOK FOR:
      If an average Urdu word tokenizes into 3+ pieces at a given vocab size,
      the vocabulary may be too small. If most words are single tokens, the
      vocabulary may be too large for the corpus size.

    Args:
        model_path    : Path to the .model file produced by train_spm_model().
        test_sentences: List of Urdu sentences for visual inspection.
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    print(f"\n  --- SPM Verification: {model_path} ---")
    for sent in test_sentences:
        pieces = sp.EncodeAsPieces(sent)
        avg_pieces_per_word = len(pieces) / max(len(sent.split()), 1)
        print(f"  Input : {sent[:60]}")
        print(f"  Tokens: {' | '.join(pieces[:15])}{'...' if len(pieces) > 15 else ''}")
        print(f"  Stats : {len(pieces)} tokens, {avg_pieces_per_word:.1f} pieces/word")
        print()


# =============================================================================
# MAIN
# =============================================================================

def run_tokenizer_training(
    vocab_sizes:     list[int] = [8_000, 16_000, 32_000],
    data_dir:        str = DATA_DIR,
    tokenizer_dir:   str = TOKENIZER_DIR,
) -> None:
    """
    Orchestrates SPM training for all requested vocabulary sizes.

    PIPELINE:
      1. Extract Urdu column from train.tsv to a plain text file
      2. For each vocab size: train BPE model
      3. Verify each model with sample tokenization

    Args:
        vocab_sizes   : List of vocab sizes to train.
        data_dir      : Directory containing processed/train.tsv.
        tokenizer_dir : Directory to write .model and .vocab files.
    """
    os.makedirs(tokenizer_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Extract Urdu corpus once (shared across all vocab sizes)
    # ------------------------------------------------------------------
    train_tsv  = os.path.join(data_dir, "processed", "train.tsv")
    urdu_corpus = os.path.join(tokenizer_dir, "urdu_corpus.txt")

    print("\n[1/2] Extracting Urdu corpus...")
    n_sentences = extract_urdu_corpus(train_tsv, urdu_corpus)

    print(f"\n  Using {n_sentences:,} Urdu sentences for BPE training.")
    print(
        f"  NOTE: SPM will use up to 2M sentences. "
        f"All {n_sentences:,} will be used."
    )

    # ------------------------------------------------------------------
    # Step 2: Train one BPE model per vocab size
    # ------------------------------------------------------------------
    print(f"\n[2/2] Training {len(vocab_sizes)} SPM models...")

    for vocab_size in vocab_sizes:
        model_prefix = os.path.join(tokenizer_dir, f"urdu_bpe_{vocab_size // 1000}k")
        train_spm_model(
            input_txt=urdu_corpus,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
        )

    # ------------------------------------------------------------------
    # Step 3: Verify models with sample sentences
    # ------------------------------------------------------------------
    sample_sentences = [
        "میں آپ سے بات کرنا چاہتا ہوں",       # I want to talk to you
        "پاکستان ایک خوبصورت ملک ہے",           # Pakistan is a beautiful country
        "اس کمپیوٹر پروگرام کو ڈاؤن لوڈ کریں", # Download this computer program
    ]

    print("\n[3/2] Verifying trained models...")
    for vocab_size in vocab_sizes:
        model_path = os.path.join(tokenizer_dir, f"urdu_bpe_{vocab_size // 1000}k.model")
        if os.path.exists(model_path):
            verify_spm_model(model_path, sample_sentences[:1])

    print("\nTokenizer training complete!")
    print(f"  Models saved to: {tokenizer_dir}/")
    print(
        f"\nNext step: fine-tune with each tokenizer:\n"
        f"  python model/train.py --variant vocab_8k\n"
        f"  python model/train.py --variant vocab_16k\n"
        f"  python model/train.py --variant vocab_32k"
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SentencePiece BPE tokenizers for vocabulary ablation."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        choices=[8000, 16000, 32000],
        help="Train a single vocab size. If omitted, all three are trained.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all three vocab sizes: 8k, 16k, 32k.",
    )
    args = parser.parse_args()

    if args.vocab_size:
        vocab_sizes = [args.vocab_size]
    else:
        vocab_sizes = [8_000, 16_000, 32_000]

    run_tokenizer_training(vocab_sizes=vocab_sizes)