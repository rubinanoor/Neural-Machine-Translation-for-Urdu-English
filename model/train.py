# =============================================================================
# model/config.py
# =============================================================================
# Experiment configuration for the NMT fine-tuning runs.
#
# WHY A CONFIG CLASS:
# Hard-coding hyperparameters in train.py means you can't reproduce a specific
# experiment without digging through code. With config files, each experiment
# is fully described by a single YAML file — you can re-run any experiment
# exactly, diff configs to see what changed, and share configs with teammates.
#
# USAGE:
#   from model.config import TrainingConfig
#   config = TrainingConfig.from_yaml("configs/vocab_16k.yaml")
#   print(config.learning_rate)
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class TrainingConfig:
    """
    All hyperparameters and paths for one fine-tuning experiment.

    Every field has a default so the class can be instantiated without
    a YAML file (useful for quick tests). YAML loading overrides defaults.
    """

    # ── Experiment identity ────────────────────────────────────────────────
    experiment_name: str = "baseline"
    # Human-readable name used for output directories and log files.
    # Set this to something descriptive like "vocab_16k_lr2e5".

    # ── Data paths ─────────────────────────────────────────────────────────
    train_path: str = "/kaggle/working/final/train.tsv"
    val_path:   str = "/kaggle/working/final/val.tsv"
    test_path:  str = "/kaggle/working/final/test.tsv"

    # ── SentencePiece model (for tokenization analysis) ────────────────────
    spm_model_path: str = "tokenizer/models/spm_bpe_16000.model"
    spm_vocab_size: int = 16000
    # NOTE: The SentencePiece model is used for ablation ANALYSIS (computing
    # fertility, OOV rate, type coverage). The actual Seq2Seq training uses
    # MarianMT's built-in tokenizer, which is already trained on Urdu-English
    # OPUS data. This is the cleanest approach because:
    # (a) MarianMT's tokenizer is already optimized for this language pair
    # (b) Replacing it would require resizing the model's embedding matrix,
    #     discarding the pre-trained weights that give us the BLEU headstart
    # The ablation report compares tokenization statistics across 8k/16k/32k.

    # ── Base model ─────────────────────────────────────────────────────────
    model_name: str = "Helsinki-NLP/opus-mt-ur-en"
    # Pre-trained MarianMT model from HuggingFace Hub.
    # Already fine-tuned on OPUS Urdu-English — we fine-tune further on
    # our cleaned domain-specific data.

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir:       str = "./checkpoints/vocab_16k"
    results_dir:      str = "./results"
    logging_dir:      str = "./logs"

    # ── Sequence lengths ───────────────────────────────────────────────────
    max_source_length: int = 128
    max_target_length: int = 128
    # 128 tokens covers > 95% of Urdu sentences in the training data.
    # Longer sequences increase GPU memory use quadratically (attention).

    # ── Training hyperparameters ───────────────────────────────────────────
    num_train_epochs:            int   = 3
    per_device_train_batch_size: int   = 32
    per_device_eval_batch_size:  int   = 64
    learning_rate:               float = 2e-5
    weight_decay:                float = 0.01
    warmup_steps:                int   = 500
    # Warmup prevents the optimizer from making large destructive updates
    # to the pre-trained weights in the first few hundred steps.

    # ── Generation parameters (for evaluation during training) ────────────
    predict_with_generate: bool = True
    generation_max_length:  int = 128
    num_beams:              int = 4
    # Beam search with width 4 is the standard for MT evaluation.
    # It's slower than greedy (beam=1) but produces measurably better BLEU.

    # ── Precision and hardware ─────────────────────────────────────────────
    fp16:       bool = True
    # fp16=True uses mixed-precision (16-bit floats) where safe, while
    # keeping 32-bit for numerically sensitive operations. This roughly
    # doubles training throughput on Kaggle's T4/A100 GPUs at no quality cost.
    # Set to False if you're running on CPU or an older GPU.

    dataloader_num_workers: int = 2

    # ── Checkpointing and evaluation ──────────────────────────────────────
    evaluation_strategy:    str = "epoch"
    save_strategy:          str = "epoch"
    save_total_limit:       int = 2
    # Only keep the 2 most recent checkpoints to avoid filling disk.
    load_best_model_at_end: bool = True
    metric_for_best_model:  str = "bleu"
    greater_is_better:      bool = True

    # ── Logging ────────────────────────────────────────────────────────────
    logging_steps:       int  = 100
    report_to:           str  = "none"
    # "none" disables W&B/TensorBoard logging — set to "tensorboard" if you
    # want training curves. Keeping it "none" by default avoids auth prompts
    # on Kaggle.

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Dataset limits (useful for debugging) ─────────────────────────────
    max_train_samples: Optional[int] = None
    max_val_samples:   Optional[int] = None
    # Set to e.g. 1000 to do a quick smoke-test run before full training.
    # Leave as None to use the full dataset.

    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """
        Loads a config from a YAML file, overriding only the fields
        that are explicitly set in the file. Fields not in the YAML
        keep their default values from this dataclass.

        This allows config files to be minimal — you only specify what
        changes between experiments, not everything.

        Args:
            yaml_path: Path to a .yaml config file.

        Returns:
            TrainingConfig instance with YAML values applied.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Validate: warn about any unrecognised keys in the YAML file.
        # This catches typos before they silently have no effect.
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        for key in yaml_data:
            if key not in valid_fields:
                print(f"  WARNING: Unknown config key '{key}' in {yaml_path} — ignoring.")

        # Filter to only known fields and create the instance
        known = {k: v for k, v in yaml_data.items() if k in valid_fields}
        return cls(**known)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Saves this config to a YAML file. Useful for logging exactly
        which config was used for a training run.

        Args:
            yaml_path: Destination path for the YAML file.
        """
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
# =============================================================================
# model/train.py
# =============================================================================
# Fine-tuning MarianMT on cleaned Urdu-English data using HuggingFace
# Seq2SeqTrainer. This script is designed to run on Kaggle GPU instances.
#
# FOR THE MID-REPORT:
#   You do NOT need to run this script to satisfy the mid-report requirement.
#   The baseline MarianMT evaluation (evaluate.py, zero-shot) is sufficient.
#   This script is included to show the complete pipeline and to enable
#   Phase 2 (vocabulary ablation) without refactoring.
#
# FOR PHASE 2 (final project):
#   Run this three times with different vocab configs:
#     python model/train.py --variant vocab_8k
#     python model/train.py --variant vocab_16k
#     python model/train.py --variant vocab_32k
#   Then evaluate each checkpoint with evaluate.py to get ablation BLEU scores.
#
# HOW TO RUN (Kaggle):
#   import sys
#   sys.path.insert(0, "/kaggle/working/urdu-en-nmt")
#   !python model/train.py --variant vocab_16k
# =============================================================================

from __future__ import annotations

import argparse
import os
import json
import numpy as np

import torch
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import sacrebleu

from model.config import AblationConfig, get_config


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_tsv_as_hf_dataset(tsv_path: str) -> Dataset:
    """
    Loads a TSV file (urdu\tenglish) as a HuggingFace Dataset object.

    HuggingFace's Seq2SeqTrainer expects datasets in the HF Dataset format.
    We convert from our TSV format here.

    Args:
        tsv_path: Path to the TSV file.

    Returns:
        HuggingFace Dataset with columns "urdu" and "english".
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {tsv_path}\n"
            "Run data/download_and_clean.py first."
        )

    urdu_sentences:    list[str] = []
    english_sentences: list[str] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                urdu_sentences.append(parts[0].strip())
                english_sentences.append(parts[1].strip())

    dataset = Dataset.from_dict({
        "urdu":    urdu_sentences,
        "english": english_sentences,
    })

    print(f"  Loaded {len(dataset):,} pairs from {tsv_path}")
    return dataset


# =============================================================================
# TOKENIZATION
# =============================================================================

def build_tokenize_fn(
    tokenizer:         MarianTokenizer,
    max_source_length: int = 128,
    max_target_length: int = 128,
):
    """
    Returns a tokenization function compatible with Dataset.map().

    BATCHED TOKENIZATION:
      We use batched=True in Dataset.map() for speed. The returned function
      processes an entire batch at once using vectorized tokenizer calls.

    SOURCE vs TARGET TOKENIZATION:
      source (Urdu)  : tokenized normally
      target (English): tokenized inside tokenizer.as_target_tokenizer()
                         This ensures MarianMT uses the correct vocab side
                         and sets the decoder_input_ids correctly.

    LABEL MASKING:
      Padding tokens in labels are replaced with -100, which tells PyTorch's
      CrossEntropyLoss to ignore those positions. Without this, the model
      would compute loss on padding tokens, which is meaningless and wastes
      gradient signal.

    Args:
        tokenizer        : Loaded MarianTokenizer.
        max_source_length: Urdu input truncation length.
        max_target_length: English output truncation length.

    Returns:
        Callable that takes a batch dict and returns tokenized tensors.
    """

    def tokenize_fn(batch: dict) -> dict:
        # Tokenize Urdu source sentences
        model_inputs = tokenizer(
            batch["urdu"],
            max_length=max_source_length,
            truncation=True,
            padding=False,   # Padding is handled by DataCollatorForSeq2Seq
        )

        # Tokenize English target sentences
        # as_target_tokenizer() is a context manager that switches the tokenizer
        # to produce decoder-side token IDs (for models with separate encoder/decoder
        # vocabularies like some OPUS-MT variants).
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["english"],
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )

        # Replace padding token id (0) with -100 in labels.
        # -100 is ignored by nn.CrossEntropyLoss, so the model does not learn
        # from padding positions.
        model_inputs["labels"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs

    return tokenize_fn


# =============================================================================
# METRIC FUNCTION (called by Trainer during eval)
# =============================================================================

def build_compute_metrics_fn(tokenizer: MarianTokenizer):
    """
    Returns a compute_metrics function for Seq2SeqTrainer.

    Seq2SeqTrainer calls compute_metrics(EvalPrediction) at the end of each
    evaluation epoch. We decode the generated token IDs and compute BLEU.

    WHY WE RETURN A CLOSURE:
      compute_metrics only receives (predictions, labels) as tensors. We need
      the tokenizer to decode them. A closure captures the tokenizer from the
      outer scope, making it available without global state.

    Returns:
        Callable that takes EvalPrediction and returns {"bleu": float}.
    """

    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred

        # If predictions are token IDs (not already decoded)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 in labels (padding) with pad_token_id for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Compute corpus-level BLEU
        bleu = sacrebleu.corpus_bleu(
            decoded_preds,
            [decoded_labels],
            tokenize="intl",
        )

        return {"bleu": round(bleu.score, 2)}

    return compute_metrics


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_fine_tuning(cfg: AblationConfig) -> None:
    """
    Full fine-tuning pipeline using HuggingFace Seq2SeqTrainer.

    STEPS:
      1. Load train/val datasets from TSV
      2. Tokenize with batched map()
      3. Initialize MarianMT from the pre-trained checkpoint
      4. Configure Seq2SeqTrainingArguments
      5. Instantiate Seq2SeqTrainer
      6. Call trainer.train()
      7. Save final model and tokenizer

    Args:
        cfg: AblationConfig with all hyperparameters and paths.
    """
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {cfg.experiment_name}")
    print(f"  Vocab size : {cfg.vocab_size:,}")
    print(f"  Epochs     : {cfg.num_train_epochs}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    print("[1/7] Loading datasets...")
    train_dataset = load_tsv_as_hf_dataset(cfg.train_tsv)
    val_dataset   = load_tsv_as_hf_dataset(cfg.val_tsv)

    # ------------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------------
    print("[2/7] Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(cfg.base_model)

    # ------------------------------------------------------------------
    # 3. Tokenize datasets
    # ------------------------------------------------------------------
    print("[3/7] Tokenizing datasets (this may take a few minutes)...")
    tokenize_fn = build_tokenize_fn(
        tokenizer,
        max_source_length=cfg.max_source_length,
        max_target_length=cfg.max_target_length,
    )

    # batched=True processes 1000 examples at a time — much faster than one-by-one.
    # num_proc=2 uses 2 CPU cores in parallel for tokenization.
    # remove_columns removes the original text columns after tokenization.
    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=2,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training set",
    )
    val_tokenized = val_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=2,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set",
    )

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    print("[4/7] Loading base model...")
    model = MarianMTModel.from_pretrained(cfg.base_model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params / 1e6:.1f}M parameters from {cfg.base_model}")

    # ------------------------------------------------------------------
    # 5. Configure training arguments
    # ------------------------------------------------------------------
    print("[5/7] Configuring training arguments...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,

        # --- Epochs and batch size ---
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        # --- Optimizer ---
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        # AdaFactor uses adaptive learning rates — better for seq2seq tasks
        # and requires less memory than AdamW. Disable scale_parameter for
        # fine-tuning (we set our own learning rate).
        optim="adafactor",

        # --- Generation during eval ---
        # predict_with_generate=True makes the trainer use model.generate()
        # (beam search) during evaluation, so we get real translation quality.
        # Without this, the trainer would use teacher-forced loss accuracy,
        # which is NOT what we want to measure.
        predict_with_generate=True,
        generation_num_beams=cfg.num_beams,
        generation_max_length=cfg.max_target_length,

        # --- Evaluation and saving ---
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,

        # --- Logging ---
        logging_steps=cfg.logging_steps,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        report_to="none",   # Set to "wandb" if you use Weights & Biases

        # --- Reproducibility ---
        seed=cfg.seed,

        # --- Mixed precision ---
        # fp16=True uses 16-bit floating point on the GPU, roughly doubling
        # training speed and halving VRAM usage at the cost of slight numeric
        # precision. Always use on Kaggle T4/P100 GPUs.
        fp16=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------------
    # 6. Instantiate Trainer
    # ------------------------------------------------------------------
    print("[6/7] Initializing Seq2SeqTrainer...")

    # DataCollatorForSeq2Seq handles dynamic padding per batch,
    # which is more memory-efficient than padding all sequences globally.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,  # Matches our tokenize_fn mask
        pad_to_multiple_of=8,     # Tensor Core alignment for mixed precision
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(tokenizer),
        callbacks=[
            # Stop training if validation BLEU does not improve for 3 evals.
            # Prevents overfitting on small fine-tuning data.
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("[7/7] Starting training...")
    train_result = trainer.train()

    # Save the final model + tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)

    # Save training metrics to results/
    os.makedirs(cfg.results_dir, exist_ok=True)
    metrics_file = os.path.join(cfg.results_dir, f"{cfg.experiment_name}_train_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "experiment": cfg.experiment_name,
                "vocab_size": cfg.vocab_size,
                "epochs":     cfg.num_train_epochs,
                "train_loss": round(train_result.training_loss, 4),
                "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 1),
            },
            f,
            indent=2,
        )

    print(f"\nTraining complete!")
    print(f"  Best model saved to : {cfg.output_dir}")
    print(f"  Train metrics       : {metrics_file}")
    print(
        f"\nNext step: evaluate with:\n"
        f"  python model/evaluate.py --variant {cfg.experiment_name.replace('ablation_', '')}"
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MarianMT on Urdu-English data."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vocab_16k",
        choices=["vocab_8k", "vocab_16k", "vocab_32k"],
        help="Which vocabulary ablation variant to train.",
    )
    args = parser.parse_args()

    cfg = get_config(args.variant)
    run_fine_tuning(cfg)
    def display(self) -> None:
        """Prints all config fields in a readable format."""
        print("\n── Training Configuration ──────────────────────────────────────")
        for key, val in asdict(self).items():
            print(f"  {key:<35}: {val}")
        print("────────────────────────────────────────────────────────────────\n")