# =============================================================================
# data/normalization.py
# =============================================================================
# Text normalization for Urdu and English sides of sentence pairs.
#
# WHY NORMALIZATION MATTERS FOR BPE:
# If the same word appears in two Unicode forms (with/without diacritics, or
# with two different code points for the same character), BPE treats them as
# completely different tokens. This wastes vocabulary slots and reduces
# coverage. Normalizing before tokenizer training AND before fine-tuning
# significantly improves token reuse and model generalization.
# =============================================================================

import re
import unicodedata


def normalize_urdu(text: str) -> str:
    """
    Normalizes Urdu text to reduce superficial variation that would cause
    the tokenizer to treat the same word as different tokens.

    NORMALIZATIONS APPLIED (in order):
    1. NFC Unicode normalization  — composes combining characters
    2. Diacritic removal          — strips harakat (short vowel marks)
    3. Zero-width char removal    — strips ZWJ / ZWNJ rendering hints
    4. Eastern → Western numerals — ٠١٢٣ → 0123
    5. Arabic ya  → Urdu ye       — U+064A → U+06CC
    6. Arabic kaf → Urdu kaf      — U+0643 → U+06A9
    7. Whitespace collapse + strip

    Args:
        text: Raw Urdu string.

    Returns:
        Normalized Urdu string, or the original value if falsy.
    """
    if not text:
        return text

    # --- 1. Unicode NFC normalization ---
    # Combines decomposed characters (base + combining mark) into their
    # precomposed single-codepoint form, ensuring consistent representation.
    text = unicodedata.normalize("NFC", text)

    # --- 2. Remove Arabic diacritics (harakat / tashkeel) ---
    # U+064B–U+065F: fatha, kasra, damma, sukun, shadda, tanwin variants, etc.
    # U+0670: Arabic letter superscript alef (dagger alef)
    # These rarely appear in normal Urdu prose but are common in Quranic text
    # (Tanzil corpus). Removing them ensures the same root word tokenizes
    # identically whether or not it carries vowel marks.
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    # --- 3. Remove zero-width characters ---
    # U+200C: Zero Width Non-Joiner (ZWNJ) — cursive rendering hint
    # U+200D: Zero Width Joiner (ZWJ)       — cursive rendering hint
    # U+200E: Left-to-Right Mark (LRM)
    # U+200F: Right-to-Left Mark (RLM)
    # These are display artifacts and must not become part of BPE tokens.
    text = re.sub(r'[\u200C\u200D\u200E\u200F]', '', text)

    # --- 4. Normalize Eastern Arabic numerals → Western Arabic numerals ---
    # Some OPUS sources use ٠١٢٣٤٥٦٧٨٩ while others use 0123456789.
    # Normalizing to Western prevents separate BPE token paths per numeral system.
    eastern_to_western = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    text = text.translate(eastern_to_western)

    # --- 5. Arabic ya → Urdu ye ---
    # U+064A (Arabic letter ya ي) vs U+06CC (Urdu letter ye ی).
    # Some sources mix the two; normalize to the Urdu-standard form.
    text = text.replace('\u064A', '\u06CC')

    # --- 6. Arabic kaf → Urdu kaf ---
    # U+0643 (Arabic letter kaf ك) vs U+06A9 (Urdu letter kaf ک).
    text = text.replace('\u0643', '\u06A9')

    # --- 7. Collapse multiple spaces and strip edges ---
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_english(text: str) -> str:
    """
    Applies light normalization to the English side of a sentence pair.

    English does not need the heavy Unicode work required for Urdu, but
    web-scraped text often carries typographic artifacts that create
    spurious token variation.

    NORMALIZATIONS APPLIED:
    1. Curly/smart quotes → straight quotes
    2. Em-dash / en-dash  → spaced hyphen
    3. Whitespace collapse + strip

    WHY NOT LOWERCASE:
    MarianMT's pre-trained tokenizer handles casing natively. Lowercasing
    would destroy capitalization signals that matter for proper nouns and
    sentence boundaries, hurting translation quality.

    Args:
        text: Raw English string.

    Returns:
        Lightly normalized English string, or the original value if falsy.
    """
    if not text:
        return text

    # --- 1. Smart quotes → straight quotes ---
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # ' '
    text = text.replace('\u201C', '"').replace('\u201D', '"')   # " "

    # --- 2. Em-dash / en-dash → spaced hyphen ---
    # Keeps punctuation simple and avoids rare Unicode tokens.
    text = text.replace('\u2014', ' - ').replace('\u2013', ' - ')

    # --- 3. Collapse whitespace and strip ---
    text = re.sub(r'\s+', ' ', text).strip()

    return text