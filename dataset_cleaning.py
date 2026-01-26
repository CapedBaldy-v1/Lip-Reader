"""
Dataset cleaning utility for Swin-VALLR.

This script mirrors the dataset-cleaning flow used in dataset papers:
- Text normalization and cleanup
- Optional n-gram language model building
- Optional transcript correction using n-gram context + edit distance

It is designed to be light-weight and dependency-free for future dataset builds.
"""

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from logging_utils import setup_logging, log_system_info, log_exception
from artifact_utils import get_useful_dir, save_json, save_text


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


LOGGER = setup_logging("dataset_cleaning")


LANGUAGE_OPTIONS = {
    "auto",
    "english",
    "spanish",
    "french",
    "arabic",
    "chinese",
    "greek"
}

BRACKETED_RE = re.compile(r"\[[^\]]*\]|\([^\)]*\)|<[^>]*>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
SPEAKER_RE = re.compile(r"^\s*[A-Z0-9_]{2,20}:\s*")


@dataclass
class QualityConfig:
    """Quality filtering configuration for label cleaning."""
    apply_filter: bool = False
    drop_duplicates: bool = False
    min_tokens: int = 1
    max_tokens: int = 60
    min_chars: int = 1
    max_chars: int = 300
    max_token_length: int = 30
    min_alpha_ratio: float = 0.6
    max_digit_ratio: float = 0.2
    max_non_alnum_ratio: float = 0.2
    max_repeat_char: int = 6
    min_unique_token_ratio: float = 0.3
    min_ngram_log_prob: Optional[float] = None
    max_samples: int = 8


@dataclass
class QualityStats:
    """Per-sample quality metrics."""
    token_count: int
    char_count: int
    alpha_ratio: float
    digit_ratio: float
    non_alnum_ratio: float
    max_repeat_run: int
    unique_token_ratio: float
    max_token_length: int
    avg_token_length: float
    avg_ngram_log_prob: Optional[float]
    score: float


@dataclass
class QualityAccumulator:
    """Aggregate quality metrics for reporting."""
    count: int = 0
    sums: Dict[str, float] = field(default_factory=dict)

    def add(self, stats: QualityStats) -> None:
        self.count += 1
        for key, value in {
            "token_count": stats.token_count,
            "char_count": stats.char_count,
            "alpha_ratio": stats.alpha_ratio,
            "digit_ratio": stats.digit_ratio,
            "non_alnum_ratio": stats.non_alnum_ratio,
            "max_repeat_run": stats.max_repeat_run,
            "unique_token_ratio": stats.unique_token_ratio,
            "max_token_length": stats.max_token_length,
            "avg_token_length": stats.avg_token_length,
            "score": stats.score
        }.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
        if stats.avg_ngram_log_prob is not None:
            self.sums["avg_ngram_log_prob"] = self.sums.get("avg_ngram_log_prob", 0.0) + float(stats.avg_ngram_log_prob)

    def averages(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        averages = {key: value / self.count for key, value in self.sums.items()}
        return averages


def _normalize_unicode_punct(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    return text


def _detect_language(text: str) -> str:
    arabic = 0
    cjk = 0
    greek = 0
    latin = 0
    spanish_markers = 0
    french_markers = 0

    for ch in text:
        code = ord(ch)
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
            arabic += 1
            continue
        if 0x4E00 <= code <= 0x9FFF:
            cjk += 1
            continue
        if 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
            greek += 1
            continue
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or (0x00C0 <= code <= 0x024F):
            latin += 1
            if code in {0x00A1, 0x00BF, 0x00D1, 0x00F1, 0x00E1, 0x00E9, 0x00ED, 0x00F3, 0x00FA, 0x00FC}:
                spanish_markers += 1
            if code in {
                0x00C0, 0x00C2, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB,
                0x00CE, 0x00CF, 0x00D4, 0x00D9, 0x00DB, 0x00DC, 0x0178,
                0x00E0, 0x00E2, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB,
                0x00EE, 0x00EF, 0x00F4, 0x00F9, 0x00FB, 0x00FC, 0x0153, 0x00E6
            }:
                french_markers += 1

    if arabic > 0:
        return "arabic"
    if cjk > 0:
        return "chinese"
    if greek > 0:
        return "greek"
    if latin > 0:
        if spanish_markers > french_markers and spanish_markers > 0:
            return "spanish"
        if french_markers > 0:
            return "french"
        return "english"
    return "auto"


def normalize_text(
    text: str,
    language: str = "auto",
    lowercase: bool = True,
    strip_bracketed: bool = False,
    strip_speaker_tags: bool = False,
    remove_urls: bool = False,
    strip_diacritics: bool = False
) -> str:
    """Normalize text by removing non-language characters and collapsing whitespace."""
    text = _normalize_unicode_punct(text)
    if strip_diacritics:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))

    if remove_urls:
        text = URL_RE.sub(" ", text)

    if strip_speaker_tags:
        text = SPEAKER_RE.sub("", text)

    if strip_bracketed:
        text = BRACKETED_RE.sub(" ", text)

    text = text.strip()

    language = language if language != "auto" else _detect_language(text)
    if lowercase:
        text = text.lower()

    if language in {"english", "spanish", "french"}:
        text = re.sub(r"[^a-z0-9' \u00C0-\u024F]+", " ", text)
    elif language == "greek":
        text = re.sub(r"[^ \u0370-\u03FF\u1F00-\u1FFF0-9']+", " ", text)
    elif language == "arabic":
        text = re.sub(r"[^ \u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9']+", " ", text)
    elif language == "chinese":
        text = re.sub(r"[^ \u4E00-\u9FFF0-9']+", " ", text)
    else:
        cleaned = []
        for ch in text:
            if ch.isalnum() or ch in " '":
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        text = "".join(cleaned)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    if not text:
        return []
    return text.split()


def _max_repeat_run(text: str) -> int:
    if not text:
        return 0
    max_run = 1
    current = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            current += 1
            if current > max_run:
                max_run = current
        else:
            prev = ch
            current = 1
    return max_run


def _average_ngram_log_prob(tokens: List[str], model: NgramModel, smoothing: float) -> Optional[float]:
    if not tokens:
        return None
    total = 0.0
    for i, token in enumerate(tokens):
        start = max(0, i - (model.order - 1))
        context = tokens[start:i]
        total += model.log_prob(context, token, smoothing=smoothing)
    return total / max(len(tokens), 1)


def compute_quality(
    raw_text: str,
    cleaned_text: str,
    tokens: List[str],
    model: Optional[NgramModel],
    quality_config: QualityConfig,
    smoothing: float
) -> Tuple[QualityStats, List[str]]:
    """Compute quality metrics and failure reasons for a sample."""
    raw = raw_text.strip()
    raw_chars = max(len(raw), 1)
    alpha_count = sum(1 for ch in raw if ch.isalpha())
    digit_count = sum(1 for ch in raw if ch.isdigit())
    alnum_count = sum(1 for ch in raw if ch.isalnum())
    non_alnum_count = max(raw_chars - alnum_count, 0)

    token_count = len(tokens)
    char_count = len(cleaned_text.replace(" ", ""))
    alpha_ratio = alpha_count / raw_chars
    digit_ratio = digit_count / raw_chars
    non_alnum_ratio = non_alnum_count / raw_chars
    max_repeat_run = _max_repeat_run(cleaned_text)
    unique_token_ratio = len(set(tokens)) / max(token_count, 1)
    max_token_length = max((len(token) for token in tokens), default=0)
    avg_token_length = sum((len(token) for token in tokens)) / max(token_count, 1)

    avg_ngram_log_prob = None
    if model is not None and tokens:
        avg_ngram_log_prob = _average_ngram_log_prob(tokens, model, smoothing=smoothing)

    score_terms = []
    if quality_config.min_alpha_ratio > 0:
        score_terms.append(min(1.0, alpha_ratio / quality_config.min_alpha_ratio))
    if quality_config.max_digit_ratio > 0:
        score_terms.append(1.0 - min(1.0, digit_ratio / quality_config.max_digit_ratio))
    if quality_config.max_non_alnum_ratio > 0:
        score_terms.append(1.0 - min(1.0, non_alnum_ratio / quality_config.max_non_alnum_ratio))
    if quality_config.min_unique_token_ratio > 0:
        score_terms.append(min(1.0, unique_token_ratio / quality_config.min_unique_token_ratio))
    score = sum(score_terms) / max(len(score_terms), 1)

    stats = QualityStats(
        token_count=token_count,
        char_count=char_count,
        alpha_ratio=alpha_ratio,
        digit_ratio=digit_ratio,
        non_alnum_ratio=non_alnum_ratio,
        max_repeat_run=max_repeat_run,
        unique_token_ratio=unique_token_ratio,
        max_token_length=max_token_length,
        avg_token_length=avg_token_length,
        avg_ngram_log_prob=avg_ngram_log_prob,
        score=score
    )

    reasons: List[str] = []
    if token_count < quality_config.min_tokens:
        reasons.append("min_tokens")
    if token_count > quality_config.max_tokens:
        reasons.append("max_tokens")
    if char_count < quality_config.min_chars:
        reasons.append("min_chars")
    if char_count > quality_config.max_chars:
        reasons.append("max_chars")
    if max_token_length > quality_config.max_token_length:
        reasons.append("max_token_length")
    if alpha_ratio < quality_config.min_alpha_ratio:
        reasons.append("alpha_ratio")
    if digit_ratio > quality_config.max_digit_ratio:
        reasons.append("digit_ratio")
    if non_alnum_ratio > quality_config.max_non_alnum_ratio:
        reasons.append("non_alnum_ratio")
    if max_repeat_run > quality_config.max_repeat_char:
        reasons.append("repeat_char")
    if unique_token_ratio < quality_config.min_unique_token_ratio:
        reasons.append("unique_token_ratio")
    if quality_config.min_ngram_log_prob is not None and avg_ngram_log_prob is not None:
        if avg_ngram_log_prob < quality_config.min_ngram_log_prob:
            reasons.append("ngram_log_prob")

    return stats, reasons


def collect_text_files(paths: Sequence[str]) -> List[Path]:
    """Collect text files from file or directory inputs."""
    files: List[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
    return files


def edit_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """Compute Levenshtein distance with optional early stopping."""
    if a == b:
        return 0
    if max_distance is not None and abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        min_in_row = i
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            min_in_row = min(min_in_row, value)
        if max_distance is not None and min_in_row > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def soundex(word: str) -> str:
    """Compute a simple Soundex key for ASCII words."""
    if not word.isascii():
        return ""
    word = re.sub(r"[^A-Za-z]", "", word).upper()
    if not word:
        return ""
    first = word[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    digits: List[str] = []
    for ch in word[1:]:
        code = mapping.get(ch, "0")
        if code != "0" and (not digits or digits[-1] != code):
            digits.append(code)
        elif code == "0":
            digits.append(code)
    digits = [d for d in digits if d != "0"]
    code = first + "".join(digits)
    return (code + "000")[:4]


@dataclass
class NgramModel:
    """Simple n-gram language model for correction scoring."""
    order: int
    unigram_counts: Dict[str, int]
    bigram_counts: Dict[Tuple[str, str], int]
    trigram_counts: Dict[Tuple[str, str, str], int]
    total_unigrams: int

    @property
    def vocab(self) -> List[str]:
        return list(self.unigram_counts.keys())

    def log_prob(self, context: List[str], token: str, smoothing: float = 1.0) -> float:
        """Compute smoothed log probability for a token with left context."""
        vocab_size = max(len(self.unigram_counts), 1)
        if self.total_unigrams == 0:
            return float("-inf")

        if self.order >= 3 and len(context) >= 2:
            ctx = (context[-2], context[-1])
            trigram = (context[-2], context[-1], token)
            trigram_count = self.trigram_counts.get(trigram, 0)
            bigram_count = self.bigram_counts.get(ctx, 0)
            prob = (trigram_count + smoothing) / (bigram_count + smoothing * vocab_size)
            return math.log(prob)

        if self.order >= 2 and context:
            bigram = (context[-1], token)
            bigram_count = self.bigram_counts.get(bigram, 0)
            unigram_count = self.unigram_counts.get(context[-1], 0)
            prob = (bigram_count + smoothing) / (unigram_count + smoothing * vocab_size)
            return math.log(prob)

        unigram_count = self.unigram_counts.get(token, 0)
        prob = (unigram_count + smoothing) / (self.total_unigrams + smoothing * vocab_size)
        return math.log(prob)

    def to_json(self) -> Dict:
        """Serialize the model to a JSON-friendly dict."""
        return {
            "order": self.order,
            "total_unigrams": self.total_unigrams,
            "unigrams": self.unigram_counts,
            "bigrams": {"\t".join(k): v for k, v in self.bigram_counts.items()},
            "trigrams": {"\t".join(k): v for k, v in self.trigram_counts.items()},
        }

    @classmethod
    def from_json(cls, data: Dict) -> "NgramModel":
        """Load a model from a JSON-friendly dict."""
        bigrams = {tuple(k.split("\t")): v for k, v in data.get("bigrams", {}).items()}
        trigrams = {tuple(k.split("\t")): v for k, v in data.get("trigrams", {}).items()}
        return cls(
            order=int(data.get("order", 3)),
            unigram_counts={k: int(v) for k, v in data.get("unigrams", {}).items()},
            bigram_counts={(k[0], k[1]): int(v) for k, v in bigrams.items()},
            trigram_counts={(k[0], k[1], k[2]): int(v) for k, v in trigrams.items()},
            total_unigrams=int(data.get("total_unigrams", 0)),
        )


def build_ngram_model(tokens: Iterable[str], order: int = 3) -> NgramModel:
    """Build an n-gram model from a stream of tokens."""
    unigrams: Counter[str] = Counter()
    bigrams: Counter[Tuple[str, str]] = Counter()
    trigrams: Counter[Tuple[str, str, str]] = Counter()

    tokens = list(tokens)
    for token in tokens:
        unigrams[token] += 1

    for i in range(len(tokens) - 1):
        bigrams[(tokens[i], tokens[i + 1])] += 1

    for i in range(len(tokens) - 2):
        trigrams[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

    return NgramModel(
        order=order,
        unigram_counts=dict(unigrams),
        bigram_counts=dict(bigrams),
        trigram_counts=dict(trigrams),
        total_unigrams=sum(unigrams.values()),
    )


def build_vocab_index(vocab: Iterable[str]) -> Dict[int, List[str]]:
    """Index vocabulary by token length for faster candidate lookups."""
    index: Dict[int, List[str]] = defaultdict(list)
    for word in vocab:
        index[len(word)].append(word)
    return index


def build_phonetic_index(vocab: Iterable[str]) -> Dict[str, List[str]]:
    """Index vocabulary by soundex key."""
    index: Dict[str, List[str]] = defaultdict(list)
    for word in vocab:
        key = soundex(word)
        if key:
            index[key].append(word)
    return index


def generate_candidates(
    word: str,
    vocab_index: Dict[int, List[str]],
    max_distance: int,
    max_candidates: int,
    use_phonetic: bool,
    phonetic_index: Optional[Dict[str, List[str]]]
) -> List[Tuple[str, int]]:
    """Generate candidate replacements within a max edit distance."""
    candidates: List[Tuple[str, int]] = []
    if not word:
        return candidates

    if use_phonetic and phonetic_index is not None:
        key = soundex(word)
        candidate_pool = phonetic_index.get(key, [])
    else:
        candidate_pool = []
        for length in range(len(word) - max_distance, len(word) + max_distance + 1):
            if length in vocab_index:
                candidate_pool.extend(vocab_index[length])

    for candidate in candidate_pool:
        dist = edit_distance(word, candidate, max_distance)
        if dist <= max_distance:
            candidates.append((candidate, dist))

    candidates.sort(key=lambda item: item[1])
    return candidates[:max_candidates]


def correct_tokens(
    tokens: List[str],
    model: NgramModel,
    vocab_index: Dict[int, List[str]],
    phonetic_index: Optional[Dict[str, List[str]]],
    max_distance: int,
    max_candidates: int,
    distance_penalty: float,
    min_score_gain: float,
    replace_known: bool,
    use_phonetic: bool,
    smoothing: float
) -> Tuple[List[str], Counter]:
    """Correct tokens using the language model and edit distance."""
    corrections: Counter[str] = Counter()
    corrected: List[str] = []
    vocab_set = set(model.unigram_counts.keys())

    for token in tokens:
        in_vocab = token in vocab_set
        if in_vocab and not replace_known:
            corrected.append(token)
            continue

        candidates = generate_candidates(
            token,
            vocab_index,
            max_distance,
            max_candidates,
            use_phonetic,
            phonetic_index
        )

        if not candidates:
            corrected.append(token)
            continue

        current_score = None
        if in_vocab:
            current_score = model.log_prob(corrected, token, smoothing=smoothing)

        best_word = token
        best_score = float("-inf")
        for candidate, dist in candidates:
            score = model.log_prob(corrected, candidate, smoothing=smoothing)
            score -= distance_penalty * dist
            if score > best_score:
                best_score = score
                best_word = candidate

        if current_score is None or best_score > current_score + min_score_gain:
            if best_word != token:
                corrections[f"{token} -> {best_word}"] += 1
            corrected.append(best_word)
        else:
            corrected.append(token)

    return corrected, corrections


def build_corpus_tokens(
    corpus_files: Sequence[Path],
    language: str,
    lowercase: bool,
    strip_bracketed: bool = False,
    strip_speaker_tags: bool = False,
    remove_urls: bool = False,
    strip_diacritics: bool = False
) -> List[str]:
    """Load and clean corpus files into a token list."""
    tokens: List[str] = []
    for path in tqdm(corpus_files, desc="Loading corpus", unit="file"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            LOGGER.warning("Failed to read corpus file: %s", path)
            continue
        cleaned = normalize_text(
            raw,
            language=language,
            lowercase=lowercase,
            strip_bracketed=strip_bracketed,
            strip_speaker_tags=strip_speaker_tags,
            remove_urls=remove_urls,
            strip_diacritics=strip_diacritics
        )
        tokens.extend(tokenize(cleaned))
    return tokens


def load_ngram_model(path: Path) -> NgramModel:
    """Load an n-gram model from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return NgramModel.from_json(data)


def save_ngram_model(model: NgramModel, path: Path) -> None:
    """Save an n-gram model to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2), encoding="utf-8")


def process_labels(
    labels_dir: Path,
    output_dir: Optional[Path],
    in_place: bool,
    dry_run: bool,
    language: str,
    lowercase: bool,
    strip_bracketed: bool,
    strip_speaker_tags: bool,
    remove_urls: bool,
    strip_diacritics: bool,
    model: Optional[NgramModel],
    max_distance: int,
    max_candidates: int,
    distance_penalty: float,
    min_score_gain: float,
    replace_known: bool,
    use_phonetic: bool,
    smoothing: float,
    quality_config: QualityConfig,
    quality_csv_path: Optional[Path]
) -> Dict:
    """Clean and optionally correct label files."""
    label_files = sorted(labels_dir.glob("*.txt"))
    total_tokens = 0
    changed_files = 0
    kept_files = 0
    filtered_files = 0
    duplicate_files = 0
    quality_reasons: Counter[str] = Counter()
    duplicate_samples: List[str] = []
    correction_counts: Counter[str] = Counter()
    original_samples: List[str] = []
    cleaned_samples: List[str] = []
    filtered_samples: List[str] = []
    quality_csv_rows = 0
    seen_texts: Dict[str, int] = {}
    kept_accumulator = QualityAccumulator()
    filtered_accumulator = QualityAccumulator()
    quality_writer = None
    quality_file = None

    if quality_csv_path is not None:
        quality_csv_path.parent.mkdir(parents=True, exist_ok=True)
        quality_file = quality_csv_path.open("w", newline="", encoding="utf-8")
        quality_writer = csv.writer(quality_file)
        quality_writer.writerow([
            "file",
            "text",
            "token_count",
            "char_count",
            "alpha_ratio",
            "digit_ratio",
            "non_alnum_ratio",
            "max_repeat_run",
            "unique_token_ratio",
            "max_token_length",
            "avg_token_length",
            "avg_ngram_log_prob",
            "score",
            "reasons"
        ])

    vocab_index = None
    phonetic_index = None
    if model is not None:
        vocab_index = build_vocab_index(model.unigram_counts.keys())
        if use_phonetic:
            phonetic_index = build_phonetic_index(model.unigram_counts.keys())

    try:
        for path in tqdm(label_files, desc="Cleaning labels", unit="file"):
            raw = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = normalize_text(
                raw,
                language=language,
                lowercase=lowercase,
                strip_bracketed=strip_bracketed,
                strip_speaker_tags=strip_speaker_tags,
                remove_urls=remove_urls,
                strip_diacritics=strip_diacritics
            )
            tokens = tokenize(cleaned)
            total_tokens += len(tokens)

            if model is not None and tokens:
                tokens, corrections = correct_tokens(
                    tokens=tokens,
                    model=model,
                    vocab_index=vocab_index,
                    phonetic_index=phonetic_index,
                    max_distance=max_distance,
                    max_candidates=max_candidates,
                    distance_penalty=distance_penalty,
                    min_score_gain=min_score_gain,
                    replace_known=replace_known,
                    use_phonetic=use_phonetic,
                    smoothing=smoothing
                )
                correction_counts.update(corrections)
                cleaned = " ".join(tokens)

            stats, reasons = compute_quality(
                raw_text=raw,
                cleaned_text=cleaned,
                tokens=tokens,
                model=model,
                quality_config=quality_config,
                smoothing=smoothing
            )
            record = {
                "file": path.name,
                "text": cleaned,
                "token_count": stats.token_count,
                "char_count": stats.char_count,
                "alpha_ratio": stats.alpha_ratio,
                "digit_ratio": stats.digit_ratio,
                "non_alnum_ratio": stats.non_alnum_ratio,
                "max_repeat_run": stats.max_repeat_run,
                "unique_token_ratio": stats.unique_token_ratio,
                "max_token_length": stats.max_token_length,
                "avg_token_length": stats.avg_token_length,
                "avg_ngram_log_prob": stats.avg_ngram_log_prob,
                "score": stats.score,
                "reasons": reasons
            }

            if quality_writer is not None:
                quality_writer.writerow([
                    record["file"],
                    record["text"],
                    record["token_count"],
                    record["char_count"],
                    f"{record['alpha_ratio']:.4f}",
                    f"{record['digit_ratio']:.4f}",
                    f"{record['non_alnum_ratio']:.4f}",
                    record["max_repeat_run"],
                    f"{record['unique_token_ratio']:.4f}",
                    record["max_token_length"],
                    f"{record['avg_token_length']:.4f}",
                    "" if record["avg_ngram_log_prob"] is None else f"{record['avg_ngram_log_prob']:.4f}",
                    f"{record['score']:.4f}",
                    ";".join(record["reasons"])
                ])
                quality_csv_rows += 1

            is_duplicate = False
            if cleaned:
                seen_texts[cleaned] = seen_texts.get(cleaned, 0) + 1
                if seen_texts[cleaned] > 1:
                    is_duplicate = True
                    if len(duplicate_samples) < quality_config.max_samples:
                        duplicate_samples.append(cleaned)

            should_filter = quality_config.apply_filter and bool(reasons)
            should_drop_duplicate = quality_config.drop_duplicates and is_duplicate

            if should_filter or should_drop_duplicate:
                filtered_files += 1
                if should_filter:
                    quality_reasons.update(reasons)
                if should_drop_duplicate:
                    duplicate_files += 1
                    quality_reasons.update(["duplicate"])
                filtered_accumulator.add(stats)
                if len(filtered_samples) < quality_config.max_samples:
                    filtered_samples.append(cleaned)
                continue

            if cleaned != raw.strip():
                changed_files += 1
                if len(original_samples) < 5:
                    original_samples.append(raw.strip())
                    cleaned_samples.append(cleaned)

            kept_files += 1
            kept_accumulator.add(stats)

            if dry_run:
                continue

            if output_dir is not None:
                output_path = output_dir / path.name
            elif in_place:
                output_path = path
            else:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(cleaned + "\n", encoding="utf-8")
    finally:
        if quality_file is not None:
            quality_file.close()

    report = {
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir) if output_dir else None,
        "in_place": in_place,
        "dry_run": dry_run,
        "language": language,
        "lowercase": lowercase,
        "num_files": len(label_files),
        "kept_files": kept_files,
        "filtered_files": filtered_files,
        "duplicate_files": duplicate_files,
        "changed_files": changed_files,
        "total_tokens": total_tokens,
        "num_corrections": sum(correction_counts.values()),
        "top_corrections": correction_counts.most_common(20),
        "quality_config": {
            "apply_filter": quality_config.apply_filter,
            "drop_duplicates": quality_config.drop_duplicates,
            "min_tokens": quality_config.min_tokens,
            "max_tokens": quality_config.max_tokens,
            "min_chars": quality_config.min_chars,
            "max_chars": quality_config.max_chars,
            "max_token_length": quality_config.max_token_length,
            "min_alpha_ratio": quality_config.min_alpha_ratio,
            "max_digit_ratio": quality_config.max_digit_ratio,
            "max_non_alnum_ratio": quality_config.max_non_alnum_ratio,
            "max_repeat_char": quality_config.max_repeat_char,
            "min_unique_token_ratio": quality_config.min_unique_token_ratio,
            "min_ngram_log_prob": quality_config.min_ngram_log_prob
        },
        "quality_reasons": quality_reasons.most_common(20),
        "quality_summary": {
            "kept": kept_accumulator.averages(),
            "filtered": filtered_accumulator.averages()
        },
        "samples": {
            "original": original_samples,
            "cleaned": cleaned_samples,
            "filtered": filtered_samples,
            "duplicate": duplicate_samples
        }
    }

    if quality_csv_path is not None:
        report["quality_csv_rows"] = quality_csv_rows
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean dataset labels and optionally correct them using an n-gram model."
    )
    parser.add_argument("--labels-dir", required=True, help="Path to labels directory containing .txt files.")
    parser.add_argument("--output-dir", help="Directory to write cleaned labels (default: none).")
    parser.add_argument("--in-place", action="store_true", help="Overwrite label files in place.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files.")

    parser.add_argument("--language", default="auto", choices=sorted(LANGUAGE_OPTIONS))
    parser.add_argument("--keep-case", action="store_true", help="Preserve original casing.")

    parser.add_argument("--ngram-model", help="Path to an existing n-gram model JSON.")
    parser.add_argument("--save-ngram-model", help="Save a newly built n-gram model to this path.")
    parser.add_argument("--corpus", action="append", default=[], help="Corpus file or folder for n-gram model.")
    parser.add_argument("--use-labels-corpus", action="store_true", help="Build n-gram model from labels.")
    parser.add_argument("--ngram-order", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--smoothing", type=float, default=1.0)

    parser.add_argument("--max-edit-distance", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--distance-penalty", type=float, default=0.5)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--replace-known", action="store_true")
    parser.add_argument("--use-phonetic", action="store_true")

    parser.add_argument("--strip-bracketed", action="store_true",
                        help="Remove bracketed tags like [noise], (laugh), <unk>.")
    parser.add_argument("--strip-speaker-tags", action="store_true",
                        help="Remove leading SPEAKER: style tags.")
    parser.add_argument("--remove-urls", action="store_true",
                        help="Remove URL-like tokens before normalization.")
    parser.add_argument("--strip-diacritics", action="store_true",
                        help="Remove diacritics for Latin-script text.")

    parser.add_argument("--apply-quality-filter", action="store_true",
                        help="Drop labels that fail quality thresholds.")
    parser.add_argument("--drop-duplicates", action="store_true",
                        help="Drop duplicate cleaned labels.")
    parser.add_argument("--save-quality-csv", action="store_true",
                        help="Save per-file quality metrics CSV.")
    parser.add_argument("--min-tokens", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--max-token-length", type=int, default=30)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.6)
    parser.add_argument("--max-digit-ratio", type=float, default=0.2)
    parser.add_argument("--max-non-alnum-ratio", type=float, default=0.2)
    parser.add_argument("--max-repeat-char", type=int, default=6)
    parser.add_argument("--min-unique-token-ratio", type=float, default=0.3)
    parser.add_argument("--min-ngram-log-prob", type=float)
    parser.add_argument("--quality-samples", type=int, default=8,
                        help="Number of sample lines to keep in report.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_system_info(LOGGER)

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and not args.in_place and not args.dry_run:
        raise ValueError("Specify --output-dir, --in-place, or --dry-run.")

    language = args.language
    if language not in LANGUAGE_OPTIONS:
        raise ValueError(f"Unsupported language: {language}")

    lowercase = not args.keep_case
    ngram_model: Optional[NgramModel] = None

    if args.ngram_model:
        model_path = Path(args.ngram_model)
        LOGGER.info("Loading n-gram model from %s", model_path)
        ngram_model = load_ngram_model(model_path)
    else:
        corpus_paths: List[str] = list(args.corpus)
        if args.use_labels_corpus:
            corpus_paths.append(str(labels_dir))

        if corpus_paths:
            corpus_files = collect_text_files(corpus_paths)
            LOGGER.info("Building n-gram model from %d files", len(corpus_files))
            tokens = build_corpus_tokens(
                corpus_files,
                language=language,
                lowercase=lowercase,
                strip_bracketed=args.strip_bracketed,
                strip_speaker_tags=args.strip_speaker_tags,
                remove_urls=args.remove_urls,
                strip_diacritics=args.strip_diacritics
            )
            ngram_model = build_ngram_model(tokens, order=args.ngram_order)
            LOGGER.info("Built n-gram model (vocab=%d)", len(ngram_model.unigram_counts))
            if args.save_ngram_model:
                save_path = Path(args.save_ngram_model)
                save_ngram_model(ngram_model, save_path)
                LOGGER.info("Saved n-gram model to %s", save_path)

    quality_config = QualityConfig(
        apply_filter=args.apply_quality_filter,
        drop_duplicates=args.drop_duplicates,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_token_length=args.max_token_length,
        min_alpha_ratio=args.min_alpha_ratio,
        max_digit_ratio=args.max_digit_ratio,
        max_non_alnum_ratio=args.max_non_alnum_ratio,
        max_repeat_char=args.max_repeat_char,
        min_unique_token_ratio=args.min_unique_token_ratio,
        min_ngram_log_prob=args.min_ngram_log_prob,
        max_samples=args.quality_samples
    )

    report_dir = get_useful_dir("dataset_cleaning", "reports")

    report = process_labels(
        labels_dir=labels_dir,
        output_dir=output_dir,
        in_place=args.in_place,
        dry_run=args.dry_run,
        language=language,
        lowercase=lowercase,
        strip_bracketed=args.strip_bracketed,
        strip_speaker_tags=args.strip_speaker_tags,
        remove_urls=args.remove_urls,
        strip_diacritics=args.strip_diacritics,
        model=ngram_model,
        max_distance=args.max_edit_distance,
        max_candidates=args.max_candidates,
        distance_penalty=args.distance_penalty,
        min_score_gain=args.min_score_gain,
        replace_known=args.replace_known,
        use_phonetic=args.use_phonetic,
        smoothing=args.smoothing,
        quality_config=quality_config,
        quality_csv_path=(report_dir / "quality_metrics.csv") if args.save_quality_csv else None
    )
    save_json(report_dir / "cleaning_report.json", report)
    save_text(report_dir / "notes.txt", "Dataset cleaning report and top corrections.")

    LOGGER.info("Cleaning complete. Report saved to %s", report_dir)
    print(f"[DatasetCleaning] Report saved to: {report_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_exception(LOGGER, "Dataset cleaning failed with an unhandled exception.")
        raise
