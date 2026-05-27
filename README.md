
---

# Urdu-English Neural Machine Translation (NMT)

### Translate Urdu sentences into English using a fine-tuned transformer model.  
### Research Question  Does BPE vocabulary size (8k / 16k / 32k) affect translation quality for morphologically rich Urdu?  
---

The complete methodology, experiments, error analysis, and discussion are available in the project repor [urdu_english_nmt_report.pdf](urdu_english_nmt_report.pdf)

---
## Results

| System | BLEU | ChrF++ | Notes |
|--------|------|--------|-------|
| IndicTrans2 (SOTA) | 30.76 | 53.00 | Published upper bound |
| MarianMT zero-shot | 25.35 | 44.86 | No fine-tuning |
| MarianMT + raw fine-tune | 21.40 | 42.55 | Raw data degrades performance |
| MarianMT + cleaned 8k BPE | 24.88 | 47.83 | **Our system** |
| MarianMT + cleaned 16k BPE | 24.88 | 47.83 | **Our system** |
| MarianMT + cleaned 32k BPE | 24.88 | 47.83 | **Our system** |

All scores: OPUS Ur→En test set (7,489 pairs), SacreBLEU tokenize=13a, ChrF++ on 0–100 scale.

**Key findings:**
1. Data cleaning recovers 3.48 BLEU points lost from naive fine-tuning
2. ChrF++ improves +2.97 above zero-shot after cleaned fine-tuning — morphological quality genuinely improves
3. BPE vocabulary size (8k/16k/32k) has no measurable effect on BLEU when using MarianMT's pretrained tokenizer
4. At the tokenization level, 16k/32k keeps technical Urdu words whole while 8k fragments them — a real segmentation finding independent of BLEU

---
## How the Three Approaches Build on Each Other

The project runs three experiments in sequence, each motivated by the 
previous one's findings.

---

### Approach 2A — Zero-shot Baseline
* File: `notebooks/approach2_baseline.ipynb`  
* Load `opus-mt-ur-en` as-is, no training, run directly on test set.  
* Establishes what a pretrained specialist model can already do before we touch anything.  
**Result:** 25.35 BLEU, 44.86 ChrF++  
**Finding:** The model is already decent. This is our ceiling to beat.

---

### Approach 2B — Naive Fine-tuning
* File: `notebooks/approach2_baseline.ipynb` (same notebook, second half)  
* Take the same model, fine-tune it for 3 epochs on 50,000 raw 
unfiltered sentence pairs from OPUS-100.  
* The obvious next step — more training should help, right?  
**Result:** 21.40 BLEU, 42.55 ChrF++ — *worse* than zero-shot  
**Finding:** Raw data hurts. The OPUS corpus is noisy: duplicate 
Quranic verses, misaligned pairs, English strings labelled as Urdu. 
Fine-tuning on this overwrites what the model already knew without 
adding useful information. This motivates building a cleaning pipeline.

---

### Approach 3 — Cleaned Fine-tuning + Vocabulary Ablation
**File:** `approach3.py` — run this on Kaggle  
* Two things at once:
1. Build a 5-stage data cleaning pipeline, deduplicate, filter noise,
   then fine-tune on the resulting 51,399 clean pairs.
2. Train SentencePiece BPE tokenizers at 8k, 16k, 32k vocab sizes and 
   analyze how Urdu words segment differently at each size.

* Approach 2B showed data quality is the bottleneck. Fix the data,
see if performance recovers. Separately, investigate whether the choice
of BPE vocabulary size — a decision every NMT practitioner makes — 
actually matters for morphologically rich Urdu.  
**Result:** 24.88 BLEU, 47.83 ChrF++  
**Finding:** Cleaning recovers most of the lost BLEU (+3.48 over 2B) 
and pushes ChrF++ above the zero-shot baseline (+2.97), meaning 
morphological translation quality genuinely improved. Vocabulary size 
makes no difference to BLEU — the pretrained tokenizer handles Urdu 
well enough that swapping vocab sizes doesn't change translation output. 
At the segmentation level however, 16k/32k keeps technical words whole 
while 8k fragments them — a real finding about Urdu tokenization 
independent of BLEU.

---

We started with a strong pretrained model (25.35 BLEU zero-shot), showed 
that blindly fine-tuning on raw data makes it worse (21.40 BLEU), then 
demonstrated that cleaning the data first recovers the loss and improves 
morphological quality (24.88 BLEU, 47.83 ChrF++). Along the way we 
investigated whether vocabulary size matters for Urdu tokenization — it 
does at the segmentation level, but not at the BLEU level when the 
pretrained tokenizer is kept. The main takeaway: for low-resource Urdu 
NMT, **data quality matters more than tokenizer configuration**.

---

## Repository Structure

```
Neural-Machine-Translation-for-Urdu-English/
│
├── approach3.py               ← MAIN SCRIPT. Runs everything end-to-end.
│                                 Data → Tokenizers → Fine-tune × 3 → Evaluate
│
├── data/
│   ├── __init__.py           
│   ├── download_and_clean.py  ← Downloads OPUS corpora, applies 5 cleaning
│   │                            filters, saves train/val/test TSV files
│   ├── cleaning_filters.py    ← The 5 filter functions (script, length,
│   │                            ratio, content, language detection)
│   ├── normalization.py       ← Urdu Unicode cleanup: harakat removal,
│   │                            ya/kaf standardization, numeral conversion
│   └── utils.py               ← TSV I/O, deduplication helpers
│
├── model/
│   ├── __init__.py            
│   ├── config.py              ← Hyperparameter dataclasses (BaselineConfig,
│   │                            AblationConfig). Read by evaluate.py + train.py
│   ├── evaluate.py            ← Loads checkpoint, runs beam-search inference,
│   │                            computes BLEU + ChrF++, saves results
│   └── train.py               ← Fine-tunes MarianMT with Seq2SeqTrainer
│
├── tokenizer/
│   ├── __init__.py            
│   └── train_spm.py           ← Trains SentencePiece BPE at 8k/16k/32k.
│                                 Used for tokenization analysis in paper
│
├── notebooks/
│   ├── approach2_baseline.ipynb  ← Baseline (25.35 BLEU) and raw
│   │                               fine-tune (21.40 BLEU). 
│   └── baseline.ipynb            ← Baseline on cleaned data.
│                                   
│
├── results/
│   └── baseline_results/
│       ├── baseline_metrics.json     ← BLEU 25.57, ChrF++ 46.65
│       ├── baseline_predictions.txt  ← Model outputs line by line
│       ├── baseline_references.txt   ← Reference translations line by line
│       └── baseline_samples.txt      ← 20 side-by-side examples
│
├── manual_get.py              ← Fallback downloader. Use only if
│                                download_and_clean.py fails on OPUS HTTP errors
├── requirements.txt           
└── README.md                  
```

---

## How to Run on Kaggle

### One-time setup

1. Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
2. Settings (right panel) → Accelerator → **GPU T4**
3. Settings → Internet → **On**
4. Paste into Cell 1 and run:

```python
import os, sys

# Install dependencies
!pip install transformers datasets sacrebleu sentencepiece sacremoses langdetect tqdm pyyaml -q

# Clone repo (public repo — no token needed)
!git clone https://github.com/rubinanoor/Neural-Machine-Translation-for-Urdu-English.git /kaggle/working/nmt

# Set working directory
repo = '/kaggle/working/nmt'
sys.path.insert(0, repo)
os.chdir(repo)

print("Working dir:", os.getcwd())
!ls
```

### Run all Approach 3 experiments

Paste into Cell 2 and run:

```python
!python approach3.py \
    --base-dir /kaggle/working/data \
    --results-dir /kaggle/working/results
```

**That's it.** One command. Runtime ~4–5 hours on T4 GPU.

What it does in sequence:
1. Downloads GNOME, TED2020, Tanzil from OPUS (~775k raw pairs)
2. Applies 5-stage cleaning → 51,399 train / 7,489 val / 7,489 test pairs
3. Trains SentencePiece BPE tokenizers at 8k, 16k, 32k
4. Prints tokenization comparison table (segmentation analysis for report)
5. Fine-tunes MarianMT 3 times — once per vocab config
6. Evaluates each model on test set
7. Saves results to `/kaggle/working/results/`

### Output files (download these after completion)

```
/kaggle/working/results/
    final_results.csv          ← BLEU and ChrF++ for all systems
    all_metrics.json           ← same in JSON
    error_analysis_8k.csv      ← 10 worst translations, 8k model
    error_analysis_16k.csv     ← 10 worst translations, 16k model
    error_analysis_32k.csv     ← 10 worst translations, 32k model
    checkpoints/               ← saved model checkpoints per vocab size
```

### Resuming after a crash

Kaggle sessions can time out. Use flags to skip completed steps:

```python
# Data already downloaded, tokenizers already trained:
!python approach3.py \
    --base-dir /kaggle/working/data \
    --results-dir /kaggle/working/results \
    --skip-data \
    --skip-tokenizer

# Run only one vocab size to test:
!python approach3.py \
    --base-dir /kaggle/working/data \
    --results-dir /kaggle/working/results \
    --skip-data \
    --skip-tokenizer \
    --vocab 8k
```

---

## Can I Run This Locally (Mac)?

| Task | Mac | Kaggle |
|------|-----|--------|
| Data pipeline (download + clean) | Possible, ~30–60 min | Recommended |
| Tokenizer training (SPM) | Yes, ~15 min | Yes |
| Model fine-tuning | No — CPU only, would take days | Yes, ~80 min/run |
| Evaluation / inference | No — too slow | Yes, ~5 min |

**Do not attempt training or evaluation locally. Use Kaggle.**

To run only the data pipeline locally for testing:

```bash
cd Neural-Machine-Translation-for-Urdu-English
pip install -r requirements.txt
python -c "
import sys; sys.path.insert(0, '.')
from data.download_and_clean import run_pipeline
run_pipeline(base_dir='./local_data')
"
```

---

## Data Sources

All from [OPUS](https://opus.nlpl.eu):

| Corpus | Domain | Raw pairs | Cleaned pairs | Retention |
|--------|--------|-----------|---------------|-----------|
| GNOME | Software UI | 11,535 | 3,821 | 33.1% |
| TED2020 | Spoken talks | 15,569 | 15,141 | 97.3% |
| Tanzil | Quranic text | 748,320 | 730,037 | 97.6% |

After global deduplication: 91.3% removed (669,943 duplicates — Tanzil contains many repeated verse translations across editions).

**Final splits:** 51,399 train / 7,489 val / 7,489 test

---

## Vocabulary Ablation

The three vocab size runs (8k/16k/32k) all use MarianMT's native pretrained tokenizer for fine-tuning. The custom SentencePiece models trained on Urdu text are used for **tokenization analysis only** — showing how Urdu words segment differently at each vocab size.

This is the correct design: replacing MarianMT's tokenizer entirely would require resizing the embedding matrix, discarding pretrained weights, and conflating tokenizer quality with embedding reinitialization effects. The tokenization comparison table (printed in Step 3) is the vocabulary ablation contribution.

Key segmentation finding: the technical word پروگرامنگ (programming) fragments into 3 pieces at 8k vocabulary but stays whole at 16k and 32k.

---

## Known Issues

**KDE4 / Ubuntu / QED return HTTP 404** — These OPUS corpora changed URL structure. Pipeline skips them and continues with GNOME, TED2020, Tanzil. Use `manual_get.py` if you specifically need them.

**All three vocab sizes produce identical BLEU** — Expected and explained. See Vocabulary Ablation section above.

**MarianMT tied weights warning** — Harmless HuggingFace warning on checkpoint load. Does not affect results.

**ChrF++ scale** — This codebase reports 0–100. Published papers (Basit et al. 2024) use 0–1. Multiply their values by 100 to compare.

**`paper size` repeated in error analysis** — Duplicate entries in OPUS-100 test set. Data quality issue in upstream corpus, not our pipeline. Discussed in paper.

---

## Environment

Tested on Kaggle T4 GPU (16GB VRAM), Python 3.12.

```
transformers>=4.30.0
datasets>=2.14.0
sacrebleu>=2.3.0
sentencepiece>=0.1.99
sacremoses
langdetect
tqdm
torch
numpy
pandas
pyyaml
```

Install:
```bash
pip install -r requirements.txt
```
