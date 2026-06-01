"""
Training and inference pipeline for the Multi-LexSum interactive case analyzer.

Mirrors the NLP pipeline in ``Data_Exploration/data_exploration.ipynb``:

  1. Legal-domain boilerplate stripping (regex)
  2. NLTK normalization (tokenize -> stopwords -> WordNet lemmatize) for the classifiers
  3. TF-IDF extractive sentence ranking -> 480-token excerpt for the summarizer
  4. Logistic Regression on TF-IDF features for `class_action_sought` and `case_type`
  5. DistilBART cascade summarization (excerpt -> long -> short -> tiny)
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path
from typing import Any, Callable

import joblib
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACTS_DIR / "sklearn_pipeline.joblib"

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
TIER_LENGTHS: dict[str, tuple[int, int]] = {
    "long":  (250, 100),
    "short": (100,  40),
    "tiny":  ( 40,  15),
}

# --------------------------------------------------------------------------- #
# NLTK + legal-text preprocessing                                             #
# --------------------------------------------------------------------------- #

def _ensure_nltk() -> None:
    """Download corpora if not already cached. Called lazily on first use."""
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


_LEMMATIZER: WordNetLemmatizer | None = None
_STOPWORDS: set[str] | None = None
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

_BOILERPLATE_PATTERNS = [
    re.compile(r"\b(?:No\.|Case)\s+\d+[\d:\-cvCV]*\b"),
    re.compile(r"\d+\s+U\.S\.C\.\s*§?\s*\d+(?:\([a-z0-9]+\))*"),
    re.compile(r"\d+\s+[A-Z]\.\s?\d?[a-z]?\s+\d+"),
    # Only strip court-name headers that appear at the START of a line (standalone
    # document headers).  The original [^\n]* was stripping the rest of the entire
    # string when the text had no newlines (e.g. short narrative example texts),
    # leaving only a 15-word fragment and bypassing BART entirely.
    re.compile(r"(?m)^(?:UNITED STATES|U\.S\.)\s+(?:DISTRICT|COURT OF APPEALS)[^\n]*", re.IGNORECASE),
    re.compile(r"Document\s+\d+(?:-\d+)?\s+Filed\s+\d{1,2}/\d{1,2}/\d{2,4}"),
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"<[^>]+>"),
]


def strip_legal_boilerplate(text: str) -> str:
    if not text:
        return ""
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Full NLTK pipeline: tokenize -> punctuation -> lowercase -> alpha -> stopwords -> lemmatize."""
    global _LEMMATIZER, _STOPWORDS
    if not text:
        return ""
    _ensure_nltk()
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
        _STOPWORDS = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [_PUNCT_RE.sub("", t) for t in tokens]
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def build_excerpt(text: str, max_tokens: int = 480) -> str:
    """Per-document TF-IDF sentence ranking → readable excerpt fitting `max_tokens`."""
    if not text or not text.strip():
        return ""
    _ensure_nltk()
    sentences = sent_tokenize(text[:300_000])
    if len(sentences) <= 1:
        return " ".join(text.split()[:max_tokens])

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(sentences)
    except ValueError:
        return " ".join(text.split()[:max_tokens])

    scores = tfidf.mean(axis=1).A1
    ranked = scores.argsort()[::-1]

    chosen, used = [], 0
    for idx in ranked:
        n = len(sentences[idx].split())
        if used + n > max_tokens:
            continue
        chosen.append(idx)
        used += n
        if used >= max_tokens:
            break

    chosen.sort()
    return " ".join(sentences[i] for i in chosen)


# --------------------------------------------------------------------------- #
# Training data builder                                                       #
# --------------------------------------------------------------------------- #

def build_train_frame_from_split(train_split: Any) -> pd.DataFrame:
    rows = []
    for ex in train_split:
        sources = ex.get("sources") or []
        full_text = "\n\n".join(sources) if sources else ""
        meta = ex.get("case_metadata") or {}
        prepped = strip_legal_boilerplate(full_text)
        rows.append(
            {
                "clean_text":          clean_text(prepped),
                "class_action_sought": meta.get("class_action_sought"),
                "case_type":           meta.get("case_type"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# DistilBART cascade summarizer                                               #
# --------------------------------------------------------------------------- #

class _Summarizer:
    """DistilBART seq2seq with a hierarchical cascade: excerpt -> long -> short -> tiny."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self._model.eval()

    def _decode(self, text: str, max_new: int, min_new: int) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        n_in = inputs["input_ids"].shape[1]
        # early_stopping=True conflicts with min_new_tokens in transformers 4.40+ for
        # short inputs — beam search can declare "all beams done" before the minimum
        # is reached.  length_penalty=2.0 additionally biases beam scoring toward
        # longer candidates so the model doesn't collapse to a single sentence.
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
            )
        decoded = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        print(
            f"[BART] in={n_in}tok  out={out.shape[1]}tok  "
            f"words={len(decoded.split())}  min_new={min_new}  "
            f"text={repr(decoded[:120])}",
            flush=True,
        )
        return decoded

    def generate_summaries(self, excerpt: str) -> dict[str, str]:
        """Cascade summarization mirroring how Multi-LexSum experts wrote the ground truth."""
        words = excerpt.split() if excerpt else []
        if len(words) < 30:
            # Text too short for the cascade — return truncated tiers directly.
            return {
                "long":  " ".join(words),
                "short": " ".join(words[:60]),
                "tiny":  " ".join(words[:20]),
            }
        long_  = self._decode(excerpt, *TIER_LENGTHS["long"])
        short_ = self._decode(long_,   *TIER_LENGTHS["short"])
        tiny_  = self._decode(short_,  *TIER_LENGTHS["tiny"])
        return {"long": long_, "short": short_, "tiny": tiny_}


# --------------------------------------------------------------------------- #
# Production analyzer                                                         #
# --------------------------------------------------------------------------- #

class CaseAnalyzer:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.lr: LogisticRegression | None = None
        self.lr_type: LogisticRegression | None = None
        self._summarizer = _Summarizer()

    def load_artifacts(self) -> bool:
        if not ARTIFACT_PATH.is_file():
            return False
        data = joblib.load(ARTIFACT_PATH)
        self.vectorizer = data["vectorizer"]
        self.lr = data["lr"]
        self.lr_type = data["lr_type"]
        return True

    def save_artifacts(self) -> None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "lr": self.lr,
                "lr_type": self.lr_type,
            },
            ARTIFACT_PATH,
        )

    def train(
        self,
        log: Callable[[str], None] | None = None,
        save: bool = True,
    ) -> None:
        def _log(msg: str) -> None:
            (log or print)(msg)

        _ensure_nltk()
        _log("Loading allenai/multi_lexsum (train split only)…")
        train_split = load_dataset(
            "allenai/multi_lexsum",
            name="v20230518",
            split="train",
            trust_remote_code=True,
        )

        _log("Building training dataframe (NLTK + legal-boilerplate cleaning)…")
        df = build_train_frame_from_split(train_split)
        del train_split

        mask = (
            df["class_action_sought"].notna()
            & df["class_action_sought"].isin(["Yes", "No"])
            & (df["clean_text"].str.len() > 50)
        )
        train_df = df[mask].copy()
        del df

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X_train = self.vectorizer.fit_transform(train_df["clean_text"])

        self.lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        self.lr.fit(X_train, train_df["class_action_sought"])
        _log(f"Trained class_action_sought LR on {len(train_df)} rows.")

        top_n = 5
        type_counts = train_df["case_type"].value_counts()
        top_types = set(type_counts.head(top_n).index)
        yt = train_df["case_type"].apply(lambda t: t if pd.notna(t) and t in top_types else "Other")

        self.lr_type = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        self.lr_type.fit(X_train, yt)
        _log(f"Trained case_type LR on {len(train_df)} rows ({len(set(yt))} classes).")

        if save:
            self.save_artifacts()
            _log(f"Saved sklearn artifacts to {ARTIFACT_PATH}")

    def ensure_ready(
        self,
        log: Callable[[str], None] | None = None,
        force_retrain: bool = False,
    ) -> None:
        if not force_retrain and self.load_artifacts():
            (log or print)(f"Loaded cached sklearn models from {ARTIFACT_PATH}")
            return
        self.train(log=log, save=True)

    def _predict_with_confidence(
        self, model: LogisticRegression, vec
    ) -> tuple[str, dict[str, float]]:
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        prob_map = {str(cls): float(p) for cls, p in zip(model.classes_, probs)}
        return str(pred), prob_map

    def analyze(self, raw_text: str) -> dict[str, Any]:
        if not self.vectorizer or not self.lr or not self.lr_type:
            raise RuntimeError("Models are not loaded; call ensure_ready() first.")
        if not raw_text or not raw_text.strip():
            return {"error": "Please enter a legal case description."}

        prepped = strip_legal_boilerplate(raw_text.strip())
        cleaned = clean_text(prepped)
        excerpt = build_excerpt(prepped)

        if len(cleaned) < 50:
            return {
                "error": (
                    "Input is too short for reliable analysis "
                    "(need at least ~50 characters of legal-case text)."
                )
            }

        vec = self.vectorizer.transform([cleaned])
        sums = self._summarizer.generate_summaries(excerpt)
        pred_action, action_probs = self._predict_with_confidence(self.lr, vec)
        pred_type, type_probs = self._predict_with_confidence(self.lr_type, vec)

        return {
            "raw_char_count": len(raw_text),
            "clean_char_count": len(cleaned),
            "clean_word_count": len(cleaned.split()),
            "excerpt_word_count": len(excerpt.split()),
            "excerpt_preview": excerpt[:500] + ("…" if len(excerpt) > 500 else ""),
            "cleaned_preview": cleaned[:500] + ("…" if len(cleaned) > 500 else ""),
            "summaries": sums,
            "class_action_sought": pred_action,
            "class_action_probs": action_probs,
            "case_type": pred_type,
            "case_type_probs": type_probs,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Train / refresh sklearn artifacts for the interactive tool.")
    p.add_argument("--force", action="store_true", help="Retrain even if artifacts exist.")
    args = p.parse_args()

    if args.force and ARTIFACT_PATH.is_file():
        ARTIFACT_PATH.unlink()

    analyzer = CaseAnalyzer()
    analyzer.ensure_ready(force_retrain=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
