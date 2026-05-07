
---

# Urdu-English Neural Machine Translation (NMT)

## Data Engineering & Pre-processing Phase

This repository contains the end-to-end data pipeline for an Urdu-to-English NMT system. Our primary research focus (**RQ1**) is investigating the impact of **BPE (Byte Pair Encoding) Vocabulary Size** on the translation quality of morphologically rich languages like Urdu.

---

##  Current Milestone
We have successfully implemented a high-performance data ingestion and cleaning pipeline. The project has transitioned from raw OPUS downloads to a refined, filtered dataset ready for tokenizer training.

### Performance Highlights
* **Total Raw Pairs Ingested:** ~775,000
* **Final Cleaned Training Set:** ~18,000 - 750,000 pairs (depending on configuration).
* **Noise Rejection:** Successfully filtered ~66% of "noisy" data in technical corpora (GNOME) while maintaining 97%+ retention in high-quality corpora (TED2020/Tanzil).


---

## Active Project Files (Current Phase)
For the current phase of the project (Data Pipeline & Baseline Evaluation), the following files are the most relevant:

### Data Processing (`/data`)
* **`download_and_clean.py`**: The main orchestrator. Manages downloading, cleaning, and the final train/val/test split.
* **`cleaning_filters.py`**: Contains heuristic "gates" (script detection, length bounds, ratio checks) to determine sentence pair quality.
* **`normalization.py`**: Handles character-level cleanup (converting Arabic/Persian Unicode variants into standard Urdu characters).
* **`utils.py`**: General utility functions for I/O, deduplication, and logging.

### Model & Evaluation (`/model`)
* **`config.py`**: Centralized configuration management for evaluation parameters, hyper-parameters, and directory paths.
* **`evaluate.py`**: Contains the logic for running inference on the test set and computing BLEU/ChrF++ metrics using MarianMT.

### Notebooks & Output
* **`notebooks/baseline.ipynb`**: The primary Kaggle execution environment used to run the pipeline and generate the baseline results.
* **`results/`**: Directory containing evaluation artifacts, including `baseline_predictions.txt` and `baseline_metrics.json`.


---


## Project Structure

```text
urdu-en-nmt/
│
├── data/
│   ├── download_and_clean.py     ← Main orchestrator: refactored for direct ingestion
│   ├── cleaning_filters.py       ← Heuristic gates (contains_urdu_script, length, ratio)
│   ├── normalization.py          ← Unicode cleanup for Urdu & English
│   └── utils.py                  ← I/O helpers (save_tsv, deduplicate, logging)
│
├── tokenizer/
│   └── train_spm.py              ← SentencePiece BPE training (Vocabulary Ablation)
│
├── model/
│   ├── train.py                  ← Main Transformer training loop
│   ├── evaluate.py               ← BLEU/ChrF++ scoring on test set
│   └── config.py                 ← Hyperparameters as a dataclass/dict
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb    
│   ├── 02_tokenizer_ablation.ipynb
│   └── baseline.ipynb.          ← Kaggle implementation for benchmarking
│
├── configs/
│   ├── vocab_8k.yaml             ← Experiment configs for 8k subwords
│   ├── vocab_16k.yaml            ← Experiment configs for 16k subwords
│   └── vocab_32k.yaml            ← Experiment configs for 32k subwords
│
├── results/
│   └── .gitkeep                  ← Metrics and model checkpoints
│
├── requirements.txt              ← Dependency list (tqdm, opustools-pkg, langdetect)
├── .gitignore                    ← Configured to ignore /raw/ data and /venv/
└── README.md
```
---





##  Setup & Execution

### 1. Environment Setup
Use a Python Virtual Environment to avoid library conflicts.
```bash
python -m venv venv
source venv/bin/activate  # Mac

pip install tqdm opustools-pkg langdetect
```

### 2. Running the Pipeline
To run the full download → clean → split process, execute from the project root:
```bash
python -m data.download_and_clean --base-dir ./my_data_test
```

Look in my_data_test/final/ for the outputs (train.tsv, urdu_train_only.txt, stats.json).

##  Technical Implementation Details

### Bypassing Library Limitations
Standard libraries like `opustools` often fail on macOS due to XML alignment errors. We solved this by implementing a **Direct Moses Ingestion** layer in `download_and_clean.py` that:
1. Targets alphabetical language pairs (e.g., `en-ur`) on the OPUS servers.
2. Streams ZIP files directly into memory to avoid file-system permission issues.
3. Decodes and pairs sentences using a robust line-by-line generator.

### Cleaning Heuristics
Each sentence pair is validated against:
* **Script Filtering**: Uses Regex to ensure the Urdu side actually contains Urdu Unicode blocks (rejects Latin-only noise).
* **Length Bounds**: Rejects "empty" or "massive" sentences that usually represent alignment glitches.
* **Cross-Split Overlap**: A safety check that deletes any training sentence that accidentally appears in our Test set, ensuring our evaluation is 100% fair.


### Datasets
Targeted: GNOME, KDE4, Ubuntu, QED, TED2020, Tanzil.

Active: GNOME, TED2020, and Tanzil. (Others bypassed due to persistent upstream XML alignment issues).

Noise Rejection: Successfully filtered ~66% of "noisy" data in technical corpora (GNOME) while maintaining 97%+ retention in high-quality corpora (TED2020/Tanzil).



---

##  Next Steps
1. Configure Custom Training Loops: Set up the Transformer training scripts using the custom 8k, 16k, and 32k tokenizers.

2. Execute Ablation Study: Train models on the custom vocabulary sizes and evaluate against the 25.57 BLEU baseline.
```

---
