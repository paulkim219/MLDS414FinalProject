"""
Training and inference pipeline for the Multi-LexSum interactive case analyzer,
aligned with Data_Exploration/exploration_paul.ipynb (sections 2–6 and Task 7).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACTS_DIR / "sklearn_pipeline.joblib"

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
LENGTH_PRESETS: dict[str, tuple[int, int]] = {
    "long": (250, 100),
    "short": (100, 40),
    "tiny": (40, 15),
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\!\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_train_frame_from_split(train_split: Any) -> pd.DataFrame:
    """Materialize only the train split rows needed for sklearn (matches notebook filters)."""
    rows = []
    for ex in train_split:
        sources = ex.get("sources") or []
        full_text = "\n\n".join(sources) if sources else ""
        meta = ex.get("case_metadata") or {}
        rows.append(
            {
                "clean_text": clean_text(full_text),
                "class_action_sought": meta.get("class_action_sought"),
                "case_type": meta.get("case_type"),
            }
        )
    return pd.DataFrame(rows)


class _Summarizer:
    """DistilBART seq2seq — same settings as the notebook."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self._model.eval()

    def _summarize(self, text: str, max_len: int, min_len: int) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_len,
                min_new_tokens=min_len,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def generate_summaries(self, text: str) -> dict[str, str]:
        trunc = text[:6000] if text else ""
        if len(trunc.split()) < 30:
            return {k: trunc for k in LENGTH_PRESETS}
        return {
            tier: self._summarize(trunc, mx, mn) for tier, (mx, mn) in LENGTH_PRESETS.items()
        }


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
            if log:
                log(msg)
            else:
                print(msg)

        _log("Loading allenai/multi_lexsum (train split only, to limit RAM) …")
        train_split = load_dataset(
            "allenai/multi_lexsum",
            name="v20230518",
            split="train",
            trust_remote_code=True,
        )

        _log("Building training dataframe …")
        df = build_train_frame_from_split(train_split)
        del train_split

        mask = (
            df["class_action_sought"].notna()
            & df["class_action_sought"].isin(["Yes", "No"])
            & (df["clean_text"].str.len() > 50)
        )
        train_df = df[mask].copy()
        del df

        # class_action_sought — fit vectorizer on this split only (notebook cell 12)
        X = train_df["clean_text"]
        y = train_df["class_action_sought"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        self.lr = LogisticRegression(max_iter=500, C=1.0)
        self.lr.fit(X_train_tfidf, y_train)
        _log(f"Trained class_action_sought LR (val accuracy check skipped). Train rows: {len(X_train)}")

        # case_type — same vectorizer, no refit (notebook cell 15)
        top_n = 5
        type_counts = train_df["case_type"].value_counts()
        top_types = set(type_counts.head(top_n).index)

        def map_case_type(t: Any) -> str:
            return t if pd.notna(t) and t in top_types else "Other"

        Xt = train_df["clean_text"]
        yt = train_df["case_type"].apply(map_case_type)

        Xt_train, Xt_val, yt_train, yt_val = train_test_split(
            Xt, yt, test_size=0.2, random_state=42, stratify=yt
        )

        Xt_train_tfidf = self.vectorizer.transform(Xt_train)
        self.lr_type = LogisticRegression(max_iter=500, C=1.0)
        self.lr_type.fit(Xt_train_tfidf, yt_train)
        _log(f"Trained case_type LR. Train rows: {len(Xt_train)}")

        if save:
            self.save_artifacts()
            _log(f"Saved sklearn artifacts to {ARTIFACT_PATH}")

    def ensure_ready(
        self,
        log: Callable[[str], None] | None = None,
        force_retrain: bool = False,
    ) -> None:
        if not force_retrain and self.load_artifacts():
            if log:
                log(f"Loaded cached sklearn models from {ARTIFACT_PATH}")
            return
        self.train(log=log, save=True)

    def analyze(self, raw_text: str) -> dict[str, Any]:
        if not self.vectorizer or not self.lr or not self.lr_type:
            raise RuntimeError("Models are not loaded; call ensure_ready() first.")

        cleaned = clean_text(raw_text.strip())
        if not cleaned:
            return {"error": "Please enter a legal case description."}

        vec = self.vectorizer.transform([cleaned])
        sums = self._summarizer.generate_summaries(cleaned)
        pred_action = self.lr.predict(vec)[0]
        pred_type = self.lr_type.predict(vec)[0]

        return {
            "cleaned_preview": cleaned[:500] + ("…" if len(cleaned) > 500 else ""),
            "summaries": sums,
            "class_action_sought": pred_action,
            "case_type": pred_type,
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
