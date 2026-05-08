# =============================================================================
# data/cleaning_filters.py
# =============================================================================
# All filter functions used in the cleaning pipeline.
# Each function accepts a (urdu, english) pair and returns bool.
#
# PIPELINE ORDER (enforced in download_and_clean.py):
#   1. contains_urdu_script   — fastest, rejects non-Urdu script immediately
#   2. passes_length_filter   — word-count arithmetic, very fast
#   3. passes_ratio_filter    — single division, very fast
#   4. passes_content_filter  — regex-based, medium speed
#   5. passes_language_filter — external library call, slowest → kept last
#
# The cheapest checks run first so expensive ones are only called on pairs
# that have already passed the easy gates.
# =============================================================================

import re
from langdetect import detect, LangDetectException


# ---------------------------------------------------------------------------
# 1. URDU SCRIPT VALIDATOR
# ---------------------------------------------------------------------------

# Unicode ranges that cover the characters used in written Urdu (Nastaliq).
# Urdu is a superset of the Arabic Unicode block plus extended/presentation forms.
URDU_UNICODE_RANGES = [
    (0x0600, 0x06FF),   # Arabic block — core Urdu letters live here
    (0x0750, 0x077F),   # Arabic Supplement
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A (ligatures)
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B (positional variants)
]


def contains_urdu_script(text: str, min_urdu_ratio: float = 0.3) -> bool:
    """
    Returns True if at least `min_urdu_ratio` fraction of the non-whitespace
    characters in `text` fall within Urdu/Arabic Unicode ranges.

    WHY THIS THRESHOLD:
    Urdu sentences routinely mix in English words, numerals, and punctuation,
    so requiring 50 %+ would incorrectly reject many valid sentences.
    0.30 (30 %) is intentionally loose — the goal is catching Roman-Urdu
    transliterations and fully-English mislabeled lines, not edge cases.

    Args:
        text           : The Urdu-side string to check.
        min_urdu_ratio : Minimum fraction of Arabic-script characters required.

    Returns:
        bool
    """
    if not text:
        return False

    urdu_char_count = 0
    total_non_space = 0

    for char in text:
        if char.isspace():
            continue
        total_non_space += 1
        cp = ord(char)
        for range_start, range_end in URDU_UNICODE_RANGES:
            if range_start <= cp <= range_end:
                urdu_char_count += 1
                break   # No need to check further ranges once matched

    if total_non_space == 0:
        return False

    return (urdu_char_count / total_non_space) >= min_urdu_ratio


# ---------------------------------------------------------------------------
# 2. LENGTH FILTER
# ---------------------------------------------------------------------------

# Per-quality thresholds. Noisy corpora use stricter limits because their
# alignments are less reliable at extreme sentence lengths.
_LENGTH_THRESHOLDS = {
    "noisy"  : {"min_ur": 5,  "max_ur": 80,  "min_en": 4,  "max_en": 80},
    "medium" : {"min_ur": 4,  "max_ur": 90,  "min_en": 3,  "max_en": 90},
    "clean"  : {"min_ur": 3,  "max_ur": 100, "min_en": 2,  "max_en": 100},
    "high"   : {"min_ur": 3,  "max_ur": 100, "min_en": 2,  "max_en": 100},
}


def passes_length_filter(urdu_text: str, english_text: str, quality: str = "medium") -> bool:
    """
    Rejects pairs that are too short (uninformative) or too long
    (likely alignment errors or merged paragraphs).

    WHY SEPARATE THRESHOLDS FOR EACH SIDE:
    Urdu is morphologically richer than English — one Urdu word can express
    what takes 2–3 English words. So a 3-word Urdu sentence is roughly
    equivalent to a 5-word English sentence; minimum word counts reflect this.

    Args:
        urdu_text   : Normalized Urdu string.
        english_text: Normalized English string.
        quality     : One of "noisy", "medium", "clean", "high".

    Returns:
        bool
    """
    t = _LENGTH_THRESHOLDS.get(quality, _LENGTH_THRESHOLDS["medium"])

    urdu_words    = len(urdu_text.split())
    english_words = len(english_text.split())

    urdu_ok    = t["min_ur"] <= urdu_words    <= t["max_ur"]
    english_ok = t["min_en"] <= english_words <= t["max_en"]

    return urdu_ok and english_ok


# ---------------------------------------------------------------------------
# 3. LENGTH RATIO FILTER
# ---------------------------------------------------------------------------

# Per-quality ratio bounds (urdu_word_count / english_word_count).
# ratio > 1  → Urdu side is longer
# ratio < 1  → English side is longer (common — Urdu is morphologically richer)
_RATIO_BOUNDS = {
    "noisy"  : (0.40, 3.0),
    "medium" : (0.35, 3.5),
    "clean"  : (0.30, 4.0),
    "high"   : (0.30, 4.0),
}


def passes_ratio_filter(urdu_text: str, english_text: str, quality: str = "medium") -> bool:
    """
    Rejects pairs where the Urdu/English word-count ratio is extreme,
    which is a strong indicator of sentence-level misalignment.

    A ratio below 0.3 suggests the English is a completely different (longer)
    text; a ratio above 4.0 suggests the Urdu is a full paragraph while
    English is just a short phrase.

    Args:
        urdu_text   : Normalized Urdu string.
        english_text: Normalized English string.
        quality     : One of "noisy", "medium", "clean", "high".

    Returns:
        bool
    """
    urdu_len    = len(urdu_text.split())
    english_len = len(english_text.split())

    # Avoid division-by-zero; empty strings should already be rejected upstream
    if english_len == 0 or urdu_len == 0:
        return False

    ratio = urdu_len / english_len
    min_ratio, max_ratio = _RATIO_BOUNDS.get(quality, _RATIO_BOUNDS["medium"])

    return min_ratio <= ratio <= max_ratio


# ---------------------------------------------------------------------------
# 4. CONTENT QUALITY FILTER
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')


def passes_content_filter(urdu_text: str, english_text: str) -> bool:
    """
    Catches specific garbage patterns that slip through length/ratio filters.

    PROBLEMS CAUGHT:
    1. Identical source and target  — teaches the model to copy, not translate.
    2. URL-heavy pairs              — scraping artifacts with no linguistic content.
    3. Low alphabetic ratio (English) — date strings, phone numbers, etc.
    4. Repeated character runs      — keyboard-spam like "aaaaaaa".
    5. Urdu side mostly ASCII       — likely Roman Urdu or mislabeled English.

    Args:
        urdu_text   : Normalized Urdu string.
        english_text: Normalized English string.

    Returns:
        bool
    """
    # 1. Source ≠ Target
    if urdu_text.strip() == english_text.strip():
        return False

    # 2. Reject English sides where > 30 % of tokens are URLs
    english_tokens = english_text.split()
    if english_tokens:
        url_count = sum(1 for tok in english_tokens if _URL_PATTERN.match(tok))
        if url_count / len(english_tokens) > 0.30:
            return False

    # 3. Reject English sides where < 40 % of characters are alphabetic
    # (catches numeric-only strings, code fragments, etc.)
    english_alpha_ratio = sum(c.isalpha() for c in english_text) / (len(english_text) + 1)
    if english_alpha_ratio < 0.40:
        return False

    # 4. Reject pairs with runs of 5+ identical characters (keyboard spam)
    if re.search(r'(.)\1{4,}', english_text) or re.search(r'(.)\1{4,}', urdu_text):
        return False

    # 5. Reject Urdu sides where > 60 % of characters are plain ASCII
    # (strongly suggests Roman Urdu or mislabeled English)
    urdu_ascii_ratio = sum(ord(c) < 128 for c in urdu_text) / (len(urdu_text) + 1)
    if urdu_ascii_ratio > 0.60:
        return False

    return True


# ---------------------------------------------------------------------------
# 5. LANGUAGE IDENTIFICATION FILTER
# ---------------------------------------------------------------------------

# Cache detection results to avoid re-running langdetect on repeated strings.
# langdetect takes ~0.5 ms per call; on 500 k pairs this saves significant time.
_lang_detect_cache: dict[str, str] = {}


def passes_language_filter(urdu_text: str, english_text: str, quality: str = "medium") -> bool:
    """
    Uses langdetect (Google's language-detection library) to verify the
    English side is actually English.

    WHY ONLY ON NOISY / MEDIUM CORPORA:
    - GNOME, KDE4, Ubuntu, Tanzil, TED2020 are already highly trustworthy.
    - langdetect gives unreliable results on very short software strings
      (< 5 words), causing false rejections on valid clean pairs.
    - Skipping it for "clean" and "high" quality tiers avoids this problem
      while still catching mislabeled pairs in the noisy corpora where the
      risk of wrong-language content is highest.

    Args:
        urdu_text   : Normalized Urdu string (not used here; script check upstream).
        english_text: Normalized English string.
        quality     : One of "noisy", "medium", "clean", "high".

    Returns:
        bool — always True for clean/high quality tiers.
    """
    if quality in ("clean", "high"):
        return True     # Trust the corpus alignment; skip the expensive call

    try:
        # Use first 100 characters as cache key to avoid hashing long strings
        cache_key = english_text[:100]
        if cache_key not in _lang_detect_cache:
            _lang_detect_cache[cache_key] = detect(english_text)

        return _lang_detect_cache[cache_key] == "en"

    except LangDetectException:
        # Raised when text is too short/uniform for reliable detection.
        # Give the pair the benefit of the doubt — it passed all other checks.
        return True