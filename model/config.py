# =============================================================================
# model/config.py
# =============================================================================
# Central configuration hub for the Urdu-English NMT project.
#
# DESIGN PHILOSOPHY:
#   All hyperparameters live in ONE place. Every other script imports from here.
#   This means you change one value and all notebooks/scripts pick it up —
#   no more hunting across files for the learning rate you last used.
#
# USAGE:
#   from model.config import BaselineConfig, AblationConfig, get_config
#
#   cfg = get_config("baseline")          # loads MarianMT baseline config
#   cfg = get_config("vocab_8k")          # loads 8k ablation config
#   cfg = get_config("vocab_16k")
#   cfg = get_config("vocab_32k")
#
# FOR THE MID-REPORT:
#   Only BaselineConfig is needed to run the MarianMT BLEU experiment.
#   AblationConfig is scaffolded here so the final project can reuse the same
#   training loop with different vocab sizes — only the YAML needs to change.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os


# =============================================================================
# PATHS — resolve relative to this file so scripts work from any directory
# =============================================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# On Kaggle: override these by setting environment variables before importing,
# or by passing a custom config dict to get_config().
#
# Example (Kaggle cell):
#   import os
#   os.environ["NMT_DATA_DIR"] = "/kaggle/working/data"
#   from model.config import get_config
# ---------------------------------------------------------------------------

DATA_DIR      = os.environ.get("NMT_DATA_DIR",      os.path.join(_PROJECT_ROOT, "data"))
TOKENIZER_DIR = os.environ.get("NMT_TOKENIZER_DIR", os.path.join(_PROJECT_ROOT, "tokenizer"))
MODEL_DIR     = os.environ.get("NMT_MODEL_DIR",     os.path.join(_PROJECT_ROOT, "model"))
RESULTS_DIR   = os.environ.get("NMT_RESULTS_DIR",   os.path.join(_PROJECT_ROOT, "results"))


# =============================================================================
# BASELINE CONFIG — MarianMT zero-shot + optional light fine-tune
# =============================================================================

@dataclass
class BaselineConfig:
    """
    Configuration for the mid-report baseline experiment.

    WHAT THIS DOES:
      Load Helsinki-NLP/opus-mt-ur-en (pre-trained on OPUS Urdu-English),
      run inference on 2,000 test sentences, and report BLEU + ChrF++ scores.
      No training required — this is purely evaluation of the pre-trained model.

    REASONING BEHIND KEY VALUES:
      • model_name_or_path : The best publicly available Urdu→English checkpoint.
                             Pre-trained by the Helsinki NLP group on the same
                             OPUS data we are using, so it's a fair baseline.
      • num_beams = 4      : Standard for NMT baselines in ACL/EMNLP papers.
                             Beam width of 1 (greedy) is 2-3 BLEU points lower.
      • max_target_length  : 128 tokens is enough for 99 %+ of our test sentences
                             (avg English sentence ≈ 20 tokens after normalization).
      • batch_size = 32    : Safe for 16 GB Kaggle GPU. Increase to 64 if using
                             a P100/T4 without OOM.
    """

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------
    experiment_name: str = "baseline_marianmt"

    # HuggingFace model hub identifier.
    # opus-mt-ur-en is Helsinki NLP's production Urdu→English MarianMT model.
    model_name_or_path: str = "Helsinki-NLP/opus-mt-ur-en"

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    # These point to the TSV files produced by download_and_clean.py.
    # Change DATA_DIR via env variable on Kaggle (see module docstring).
    test_tsv:  str = os.path.join(DATA_DIR, "processed", "test.tsv")
    train_tsv: str = os.path.join(DATA_DIR, "processed", "train.tsv")
    val_tsv:   str = os.path.join(DATA_DIR, "processed", "val.tsv")

    # ------------------------------------------------------------------
    # Evaluation settings
    # ------------------------------------------------------------------
    # Number of test sentences to evaluate on for the mid-report.
    # 2,000 is statistically stable for BLEU (variance < 0.3 points at this N).
    eval_samples: int = 2_000

    # Beam search width. Width-4 is the NMT community standard for fair comparison.
    num_beams: int = 4

    # Hard cap on generated sequence length.
    # If your test set has very long sentences, raise this to 256.
    max_source_length: int = 128
    max_target_length: int = 128

    # How many pairs to send to the GPU at once during inference.
    # Does NOT affect BLEU — only speed. Reduce if you get CUDA OOM.
    batch_size: int = 32

    # Random seed used when sampling 2k rows from the full test set.
    seed: int = 42

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    results_dir:    str = RESULTS_DIR
    predictions_file: str = os.path.join(RESULTS_DIR, "baseline_predictions.txt")
    references_file:  str = os.path.join(RESULTS_DIR, "baseline_references.txt")
    metrics_file:     str = os.path.join(RESULTS_DIR, "baseline_metrics.json")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    # "cuda" on Kaggle GPU, "cpu" for local testing. Auto-detected in evaluate.py.
    device: str = "auto"


# =============================================================================
# ABLATION CONFIG — fine-tuned variants with different BPE vocab sizes
# =============================================================================

@dataclass
class AblationConfig:
    """
    Configuration for the vocabulary-size ablation study (Phase 2 / final project).

    RESEARCH QUESTION:
      Does increasing the SentencePiece BPE vocabulary from 8k → 16k → 32k
      improve translation quality on Urdu→English?

    HYPOTHESIS:
      Urdu is morphologically rich. A larger vocabulary may capture more
      Urdu morphemes as single tokens, reducing fragmentation and helping
      the encoder represent complex words more accurately.
      However, larger vocabularies require more training data to achieve
      good coverage. We expect 16k to outperform 8k, with diminishing
      returns (or regression) at 32k given our ~500k training pairs.

    NOTE FOR MID-REPORT:
      You do NOT need to run this config for the mid-report.
      The baseline MarianMT evaluation (BaselineConfig) is sufficient.
      This is scaffolded here so Phase 2 requires zero refactoring.
    """

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------
    experiment_name: str = "ablation_vocab_16k"   # override per run

    # We fine-tune FROM the same MarianMT checkpoint to isolate the effect
    # of tokenizer vocabulary size, keeping all other variables constant.
    base_model: str = "Helsinki-NLP/opus-mt-ur-en"

    # ------------------------------------------------------------------
    # Tokenizer (SPM) settings — trained in tokenizer/train_spm.py
    # ------------------------------------------------------------------
    # Valid values: 8000, 16000, 32000
    vocab_size: int = 16_000

    # SentencePiece model produced by train_spm.py
    spm_model_prefix: str = os.path.join(TOKENIZER_DIR, "urdu_bpe_16k")

    # ------------------------------------------------------------------
    # Data paths (same as baseline)
    # ------------------------------------------------------------------
    train_tsv: str = os.path.join(DATA_DIR, "processed", "train.tsv")
    val_tsv:   str = os.path.join(DATA_DIR, "processed", "val.tsv")
    test_tsv:  str = os.path.join(DATA_DIR, "processed", "test.tsv")

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    num_train_epochs: int = 3

    # Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
    # For a 16 GB Kaggle GPU:
    #   32 × 4 = 128 effective batch size → stable training signal
    per_device_train_batch_size:  int = 32
    per_device_eval_batch_size:   int = 32
    gradient_accumulation_steps:  int = 4

    # AdaFactor is recommended for seq2seq fine-tuning over AdamW:
    # it uses less memory and is less sensitive to learning rate choice.
    # Learning rate 5e-5 is a safe default for MarianMT fine-tuning.
    learning_rate:  float = 5e-5
    warmup_steps:   int   = 200
    weight_decay:   float = 0.01

    # ------------------------------------------------------------------
    # Generation settings (used during eval)
    # ------------------------------------------------------------------
    num_beams:          int = 4
    max_source_length:  int = 128
    max_target_length:  int = 128

    # ------------------------------------------------------------------
    # Saving & evaluation
    # ------------------------------------------------------------------
    # Save the best model by validation BLEU, not by loss.
    # BLEU correlates better with human translation quality.
    metric_for_best_model: str = "bleu"
    save_total_limit:      int = 2    # Keep only best + most recent checkpoint

    output_dir:    str = os.path.join(MODEL_DIR, "checkpoints", "ablation_vocab_16k")
    results_dir:   str = RESULTS_DIR
    logging_steps: int = 100
    eval_steps:    int = 500

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = "auto"


# =============================================================================
# FACTORY FUNCTION — convenience accessor
# =============================================================================

_CONFIG_REGISTRY: dict[str, type] = {
    "baseline": BaselineConfig,
    "vocab_8k":  AblationConfig,
    "vocab_16k": AblationConfig,
    "vocab_32k": AblationConfig,
}

# Vocab-size overrides for each ablation variant
_VOCAB_OVERRIDES: dict[str, dict] = {
    "vocab_8k":  {"vocab_size": 8_000,  "experiment_name": "ablation_vocab_8k",
                  "spm_model_prefix": os.path.join(TOKENIZER_DIR, "urdu_bpe_8k"),
                  "output_dir": os.path.join(MODEL_DIR, "checkpoints", "ablation_vocab_8k")},
    "vocab_16k": {"vocab_size": 16_000, "experiment_name": "ablation_vocab_16k",
                  "spm_model_prefix": os.path.join(TOKENIZER_DIR, "urdu_bpe_16k"),
                  "output_dir": os.path.join(MODEL_DIR, "checkpoints", "ablation_vocab_16k")},
    "vocab_32k": {"vocab_size": 32_000, "experiment_name": "ablation_vocab_32k",
                  "spm_model_prefix": os.path.join(TOKENIZER_DIR, "urdu_bpe_32k"),
                  "output_dir": os.path.join(MODEL_DIR, "checkpoints", "ablation_vocab_32k")},
}


def get_config(variant: str = "baseline") -> BaselineConfig | AblationConfig:
    """
    Returns the config object for the requested experiment variant.

    Args:
        variant: One of "baseline", "vocab_8k", "vocab_16k", "vocab_32k".

    Returns:
        Populated config dataclass instance.

    Example:
        cfg = get_config("baseline")
        print(cfg.model_name_or_path)   # Helsinki-NLP/opus-mt-ur-en
    """
    if variant not in _CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Valid options: {list(_CONFIG_REGISTRY.keys())}"
        )

    cfg = _CONFIG_REGISTRY[variant]()

    # Apply per-variant overrides for ablation experiments
    if variant in _VOCAB_OVERRIDES:
        for attr, val in _VOCAB_OVERRIDES[variant].items():
            setattr(cfg, attr, val)

    return cfg