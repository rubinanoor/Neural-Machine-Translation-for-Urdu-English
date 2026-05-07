
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
│   ├── 01_data_pipeline.ipynb    ← Kaggle implementation: calls data/ scripts
│   ├── 02_tokenizer_ablation.ipynb
│   └── 03_training.ipynb
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

##  Project Navigation & File Guide

### Core Logic
* **`data/download_and_clean.py`**: The main orchestrator. It manages the loop through corpora, calls the downloader, triggers cleaning, and performs the final train/val/test split.
* **`data/cleaning_filters.py`**: Contains the heuristic "gates." This is where we define what makes a "good" sentence pair (e.g., length ratios, language detection, and script validation).
* **`data/normalization.py`**: Handles character-level cleanup. It converts various Arabic/Persian Unicode variants into standard Urdu characters to reduce vocabulary sparsity.

### Utilities & Data
* **`manual_get.py`**: A standalone helper script. Use this if the automated `opustools` library fails; it fetches Moses-format files directly via HTTP.
* **`my_data_test/final/`**: **Look here for the results.** 
    * `train.tsv`: The primary training file.
    * `urdu_train_only.txt`: Used specifically for training the SentencePiece tokenizer.
    * `stats.json`: Automated report for the Mid-Report (contains counts and retention %).



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
Using corpuses:  "GNOME","KDE4", "Ubuntu", "QED","TED2020", "Tanzil".

Currently only GNOME, TED2020 and Tanzil are working. the rest have having issues because of XML alignment issues with opustools.



---

##  Next Steps
1. **Upload** the contents of `my_data_test/final/` to Kaggle.
2. **Train Tokenizers**: Generate `8k`, `16k`, and `32k` BPE models.
3. **Begin Training**: Baseline Transformer-Base model.
```

---
