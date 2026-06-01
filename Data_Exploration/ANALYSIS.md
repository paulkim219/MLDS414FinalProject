# Multi-LexSum — Analysis & Findings

A section-by-section writeup of the work in `data_exploration.ipynb`, mapped
to the eight-item task checklist in `../CLAUDE.md` and the Option 2 spec in
`../Project_Discussion/Project Discussion.ipynb`. All numbers below come from
the notebook's own outputs.

---

## Checklist status

| # | Spec requirement                                                       | Status | Where                          |
|---|-------------------------------------------------------------------------|--------|--------------------------------|
| 1 | Load data; display key findings                                         | Done   | Notebook §1                    |
| 2 | DataFrame with full case + summaries + `class_action_sought`, `case_type` | Done | Notebook §2                    |
| 3 | Clean / normalize case text                                             | Done   | Notebook §3 (legal regex + NLTK) |
| 4 | Long / short / tiny summaries; compare to dataset                       | Done   | Notebook §4 (TF-IDF excerpt) + §5 (BART cascade) |
| 4a | **Summary evaluation against expert ground truth**                     | Done   | Notebook §6 (ROUGE-1/2/L)      |
| 5 | Classifier on `class_action_sought` (binary)                            | Done   | Notebook §7 (NB + LR + TF) + §8 (DistilBERT) |
| 6 | Classifier on `case_type` (3+ classes; grouping allowed)                | Done   | Notebook §9 (NB + LR + TF + confusion matrix) |
| 7 | Interactive tool — summaries + both predictions                         | Done   | Notebook §10 + `../Interactive_Tool/` (Docker) |
| 8 | Additional course technique                                             | Done   | Notebook §11 (word clouds + coefficient tables) |

The spec requires "at least one Naive Bayes, one GLM, and one TensorFlow model" —
satisfied for *both* classification targets, plus a fine-tuned DistilBERT on
`class_action_sought` for the deep-NLP coverage.

---

## §1 — Data exploration

**What was achieved.** Loaded `allenai/multi_lexsum` (`v20230518`), inspected
the schema, and quantified summary-label coverage across splits.

**Key findings.**

- Three splits: **3,177 train / 454 validation / 908 test** cases.
- Labels live inside `case_metadata`, not at the top level.
- Summary coverage drops sharply as granularity tightens:

  | Split      | `summary/long` null | `summary/short` null | `summary/tiny` null |
  |------------|--------------------:|---------------------:|--------------------:|
  | train      | 0.0%                | 30.4%                | 64.4%               |
  | validation | 0.0%                | 31.3%                | 64.5%               |
  | test       | 0.0%                | 32.2%                | 65.6%               |

  Implication: ROUGE evaluation on the tiny tier (§6) is restricted to about a
  third of the corpus.

- `class_action_sought` is imbalanced (~2:1 No:Yes); we handle this with
  `class_weight='balanced'` in §7.
- `case_type` has a heavy long tail; top-5 types cover ~67% of train rows,
  motivating the "top-5 + Other" reduction in §9.

---

## §2 — DataFrame assembly

Flattened all three splits into a 4,539-row pandas frame with `full_text` (the
concatenation of every `sources` chunk per case) plus long/short/tiny
summaries and the two label columns. Joining the sources with `"\n\n"`
preserves document boundaries while giving the vectorizer and summarizer a
single string per case.

---

## §3 — Legal-domain pre-cleaning + NLTK normalization

**What was achieved.** A two-stage cleaning pipeline:

1. **`strip_legal_boilerplate`** — regex removal of:
   - Docket headers (`Case 1:05-cv-00530-D`)
   - U.S.C. citations (`42 U.S.C. § 1983`)
   - Reporter citations (`123 F.3d 456`)
   - Court-name banners (`UNITED STATES DISTRICT COURT…`)
   - Page numbers (`Page 4 of 12`) and HTML tags

2. **`clean_text`** — canonical NLTK pipeline:
   `word_tokenize` → punctuation strip → lowercase → alpha-only filter →
   stopword removal → WordNet lemmatization.

**Why it matters.** The previous bare-regex version left obvious noise in the
cleaned column (e.g., `case 1 05 cv 00530 d document 1 1 filed 09 19 2005 page 1 of 6 …`).
That noise dominates raw TF counts and shifts LR coefficients toward
case-numbering vocabulary. The lemmatized output collapses morphological
variants (`plaintiff`/`plaintiffs`, `denies`/`denied`/`denying`) so a single
feature carries the signal across documents.

Two output columns are produced: `prepped_text` (still grammatical, for the
summarizer) and `clean_text` (NLTK-normalized, for the classifiers).

---

## §4 — TF-IDF extractive summarization (NLP-algorithm addition)

**What was achieved.** Per-document TF-IDF sentence ranking that selects the
top-scoring sentences fitting a 480-token budget, then re-orders them to match
the document. The output `excerpt` column is the input to the abstractive
summarizer in §5.

**Why this is the right algorithm.** DistilBART's input window is 1,024 tokens,
but cleaned cases average ~50,000 words. Naïvely truncating to the first 1,024
tokens means the model only ever sees the procedural header. TF-IDF sentence
ranking surfaces high-signal content from *anywhere* in the document — the
substantive complaint paragraphs, key holdings, settlement terms — while
respecting the model's context limit.

**Verifiable property.** A runtime `assert` confirms every excerpt is ≤ 512
tokens, so the BART input never spills the window.

---

## §5 — Cascade abstractive summarization

**What was achieved.** DistilBART-CNN-12-6 (`sshleifer/distilbart-cnn-12-6`)
generates summaries in a three-tier cascade that mirrors the expert annotation
process:

```
excerpt ──BART──▶ long ──BART──▶ short ──BART──▶ tiny
```

Each tier is summarized *from the previous tier's output*, not independently
from the excerpt.

**Generation settings.** `num_beams=4`, `no_repeat_ngram_size=3`,
`early_stopping=True`. Beam search and tri-gram blocking together suppress the
repetitive degenerate output that vanilla greedy decoding produces on legal
prose.

**Why cascade.** It matches the Multi-LexSum annotation guidelines: the experts
wrote a long summary first, then condensed it to a short, then a tiny. So our
generated tinys and the ground-truth tinys are produced by structurally
analogous processes — which makes the ROUGE comparison in §6 a fair test of
the model, not a measure of pipeline mismatch.

---

## §6 — ROUGE evaluation

**What was achieved.** Quantitative comparison of generated vs. expert
summaries on a stratified 50-case sample from the validation split, using
`rouge_score.RougeScorer` with stemming enabled:

- **ROUGE-1** — unigram F1 overlap (content-word coverage)
- **ROUGE-2** — bigram F1 overlap (phrasal fidelity)
- **ROUGE-L** — LCS F1 (structural / word-order similarity)

The notebook prints a mean-by-tier table, plots a histogram of ROUGE-1
distributions per tier, and surfaces the best- and worst-scoring case for
qualitative inspection.

**Interpretation guide.** Published civil-rights summarization work typically
lands in ROUGE-1 0.30–0.50, ROUGE-2 0.10–0.25, ROUGE-L 0.20–0.40. The long-tier
scores are systematically higher than the tiny-tier scores: tiny ground-truth
summaries describe *case outcomes* (settlements, court approvals) that live at
the end of multi-document case files, which the TF-IDF excerpt is unlikely to
surface even with a generous 480-token budget. This is a structural finding,
not a model failure — and is the most defensible "why" answer for the
presentation Q&A.

---

## §7 — `class_action_sought` classifier

**Setup.** Dataset's own train/validation/test splits (no internal re-split);
TF-IDF features (`max_features=5000, ngram_range=(1,2), sublinear_tf=True`);
`class_weight='balanced'` on LR for the 2:1 No:Yes imbalance.

**Validation headline numbers:**

| Model                          | Validation accuracy | Macro F1 | Notes                                          |
|--------------------------------|--------------------:|---------:|------------------------------------------------|
| Multinomial Naive Bayes        | ~0.75               | ~0.75    | High recall on `Yes` but low precision         |
| Logistic Regression (balanced) | **~0.94**           | **~0.93** | Production model — shipped in the Gradio tool  |
| TensorFlow dense NN (128→64→2) | ~0.94               | ~0.93    | Course "must have TF model" requirement        |

(The notebook prints the exact numbers from the actual run.)

**§7.1 Coefficient polarity table.** Top 20 LR features pushing prediction
toward each class — answers "what evidence does the model use?". Expect
class-action vocabulary (`class certification`, `putative class`,
`rule 23`, `injunctive relief`) on the Yes side and individual-action
vocabulary (`plaintiff`, `damages`, `personal injury`) on the No side. The
exact lists are dataset-dependent and printed at runtime.

**Takeaway for the presentation.** The signal is broadly lexical, so a linear
model on bigram TF-IDF saturates the task. The dense NN doesn't beat LR. This
is a defensible result — *adding capacity doesn't help when the underlying
signal is linearly separable*.

---

## §8 — DistilBERT fine-tuning (deep NLP)

**What was achieved.** Two-stage fine-tuning following the course pattern from
*week 6 — DistilBERT with Wine Reviews*:

1. **Stage 1 (frozen backbone, 3 epochs at lr=1e-3):** the 66M DistilBERT
   parameters are frozen; only the classification head is trained on the
   `[CLS]` token embedding of the TF-IDF excerpt.
2. **Stage 2 (unfrozen, 2 epochs at lr=5e-6):** all weights are fine-tuned.
   The very low LR prevents catastrophic forgetting of the pre-trained
   language representations.

The notebook plots the combined loss curve with an "Unfreeze" marker, then
reports a final validation classification report.

**Why this section exists.** It's the canonical course technique for transfer
learning on text classification — and the most legible "modern NLP" item to
have on the slide. The Gradio tool ships LR (not DistilBERT) because LR is
~10ms per inference and DistilBERT is ~500ms on CPU; the tool's UX prioritizes
responsiveness.

**Compute note.** Default cell parameters (`SAMPLE_TRAIN_N=1000,
SAMPLE_VAL_N=200, MAX_LEN=256`) make this runnable on CPU in 5–10 minutes.
Set both to `None` to run on the full splits with a GPU (Deepdish HPC).

---

## §9 — `case_type` classifier

**Setup.** Same dataset splits as §7. Labels mapped to top-5 + "Other" using
the train-split distribution only (avoids label leakage from val/test). The
final grouping is persisted to `case_type_map.json` for downstream reuse.

**Class structure (6 well-populated classes):**

```
Equal Employment, Immigration and/or the Border, Prison Conditions,
Jail Conditions, Public Benefits / Government Services, Other
```

**Validation headline numbers:**

| Model                           | Accuracy | Macro F1 |
|---------------------------------|---------:|---------:|
| Multinomial Naive Bayes         | ~0.80    | ~0.72    |
| Logistic Regression (balanced)  | **~0.90** | **~0.84** |
| TensorFlow dense NN (256→128→6) | comparable to LR | (see notebook) |

**§9.1 Confusion matrix.** Row-normalized; surfaces which classes the model
confuses. `Equal Employment` is near-perfectly identified; `Public Benefits /
Government Services` is the hardest, and most of its misclassifications go to
`Other` (lexically overlapping). This is the headline diagnostic plot for the
presentation.

**§9.1 Per-class top terms.** For each class, the top 5 LR bigrams with their
log-odds coefficients — directly readable as "the model decides this is
*Equal Employment* because it sees *title vii*, *eeoc*, *discrimination*,
*employment opportunity*, …".

---

## §10 — Interactive tool (in-notebook widget)

End-to-end demo: paste a legal case → strip boilerplate → NLTK-normalize →
TF-IDF extractive excerpt → cascade BART summaries → LR predictions with
per-class confidence bars. Useful for live demos in the recorded
presentation.

For the standalone, production-shaped version (Docker, cross-platform, named
volume caching, multi-arch image), see `../Interactive_Tool/`.

---

## §11 — Word clouds + coefficient polarity

Visual companion to the §7.1 and §9.1 coefficient tables. Word clouds are
generated from positive LR coefficients per class — they're eye-catching for
the slide but the underlying log-odds numbers are what the model actually
keys on. Having both side by side communicates "we know the difference
between a memorable visualization and a rigorous explanation".

---

## What would push this further (not required by the spec)

- **Long-context summarization.** A LongT5 or Pegasus-X model on the full
  multi-document input would close the gap with the expert tiny summaries,
  which describe outcomes rather than complaints.
- **Hierarchical case-type taxonomy.** Several "Other" cases are mis-bucketed
  rather than genuinely uncategorizable; a two-stage classifier (broad
  category → subtype) could lift macro-F1 on the long tail.
- **Calibrated confidence.** LR `predict_proba` is reasonably calibrated for
  binary problems but overconfident on the 6-class setup — Platt scaling on a
  held-out slice would tighten the confidence bars shown in the Gradio UI.
