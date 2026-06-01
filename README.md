# MLDS414 Final Project — Multi-LexSum Civil-Rights Case Analyzer

Northwestern University, MLDS 414 Text Analytics (Spring 2025).
Group project for **Option 2: Summarization of Civil Rights Lawsuits**, built on the
[allenai/multi_lexsum](https://huggingface.co/datasets/allenai/multi_lexsum) dataset
(9,280 expert-authored summaries of civil-rights legal cases).

The repository delivers:

1. End-to-end NLP pipeline (exploration, cleaning, summarization, ROUGE evaluation,
   classification) in a single notebook.
2. A standalone web interactive tool (Gradio) that runs the same pipeline on user input.
3. A cross-platform Docker image so the tool runs identically on Windows and macOS.

---

## NLP techniques applied

| Stage | Technique | Course module |
|---|---|---|
| Preprocessing | Legal-domain regex (dockets, U.S.C. / reporter citations, court headers) | Applied NLP |
| Preprocessing | NLTK `word_tokenize` → stopword removal → WordNet lemmatization | Week 3 — Text Normalization with NLTK |
| Extractive summarization | Per-document **TF-IDF sentence ranking** (480-token budget) | Week 4 — TF-IDF |
| Abstractive summarization | **DistilBART-CNN-12-6**, beam search + tri-gram blocking, cascade decoding | Week 6 — Summarization with T5/BART |
| Evaluation | **ROUGE-1 / ROUGE-2 / ROUGE-L** F1 against expert ground truth | Applied NLP |
| Classification | **Multinomial Naive Bayes** + TF-IDF (baseline, both targets) | Week 4 — Naive Bayes |
| Classification | **Logistic Regression** + TF-IDF, balanced class weights, coefficient inspection | Week 4 — Sentiment Analysis |
| Classification | **TensorFlow dense NN** on TF-IDF (course "must have a TF model" requirement) | Week 5 — Keras |
| Classification | **DistilBERT** fine-tuning (frozen-head → end-to-end, two stage) | Week 6 — DistilBERT |
| Visualization | Confusion matrix; word clouds + log-odds coefficient tables | Course-extra technique |

---

## Repository structure

```
MLDS414FinalProject/
├── README.md                        # this file
├── CLAUDE.md                        # repo guidance for the Claude Code agent
├── Project_Discussion/              # course spec (read-only reference)
│   └── Project Discussion.ipynb
├── Data_Exploration/                # all modeling and EDA
│   ├── data_exploration.ipynb       # main notebook — all 11 sections
│   ├── ANALYSIS.md                  # written summary of what each section achieves
│   └── case_type_map.json           # documented case_type grouping (written at runtime)
└── Interactive_Tool/                # standalone web tool (Task 7 deliverable)
    ├── app.py                       # Gradio UI (cascade summaries + confidence bars)
    ├── case_analyzer.py             # NLTK + TF-IDF excerpt + LR + DistilBART pipeline
    ├── requirements.txt
    ├── Dockerfile                   # multi-arch (amd64 + arm64) image with NLTK pre-cached
    ├── .dockerignore
    ├── docker-compose.yml           # one-command launch
    └── artifacts/                   # cached sklearn pipeline (created on first run)
```

---

## Architecture

```
                ┌────────────────────────────────────────┐
                │  HuggingFace: allenai/multi_lexsum     │
                │  (train / validation / test splits)    │
                └──────────────────┬─────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │  strip_legal_boilerplate()               │
              │    dockets, U.S.C. / F.3d cites, headers │
              └──────────────────┬───────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   ┌──────────────────────┐               ┌──────────────────────┐
   │ NLTK normalize       │               │ TF-IDF sentence rank │
   │  word_tokenize       │               │   per-doc TF-IDF →   │
   │  stopwords, lemmas   │               │   top-N to 480 toks  │
   │ → clean_text         │               │ → excerpt            │
   └──────────┬───────────┘               └──────────┬───────────┘
              ▼                                       ▼
   ┌──────────────────────┐               ┌──────────────────────┐
   │ TF-IDF (corpus, 1-2 │                │ DistilBART cascade   │
   │  grams, 5k feats)   │                │  excerpt → long →    │
   └──────────┬──────────┘                │  short → tiny        │
              │                            └──────────┬───────────┘
   ┌──────────┴──────────┐                            ▼
   ▼                     ▼                  ┌────────────────────┐
LR + NB +  TF Dense    LR + NB + TF Dense  │  ROUGE-1/2/L vs    │
DistilBERT             (case_type)         │  expert ground     │
(class_action)                              │  truth (n=50)      │
        │                       │           └────────────────────┘
        └────────┬──────────────┘
                 ▼
   joblib artifact → Interactive_Tool/artifacts/sklearn_pipeline.joblib
                 │
                 ▼
   Gradio UI: paste case → predictions w/ confidence + cascade summaries
```

---

## Quick start

### Option A: Docker (recommended, identical on Windows + macOS)

Prerequisite: [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

From the repo root:

```bash
docker compose -f Interactive_Tool/docker-compose.yml up --build
```

Then open <http://localhost:7860> in your browser.

What happens on first launch:

1. The image is built (~3 GB; downloads `python:3.11-slim`, Python deps, and
   pre-caches the NLTK corpora `punkt / stopwords / wordnet / omw-1.4`).
2. The container boots, then trains the sklearn pipeline from the dataset
   (~3–5 min) and caches it to a named volume (`sklearn-artifacts`).
3. On first analysis request, DistilBART (~1.2 GB) is downloaded into the
   `hf-cache` named volume.

Subsequent `docker compose up` runs reuse both caches and start in seconds.

The image base (`python:3.11-slim`) ships as a multi-arch manifest, so Docker
pulls the correct architecture automatically — `linux/amd64` on Windows
(Docker Desktop / WSL2 backend) and `linux/arm64` on Apple-Silicon Macs.

To stop:

```bash
docker compose -f Interactive_Tool/docker-compose.yml down
```

### Option B: Native Python

Requires Python 3.11+.

```bash
# 1. Set up a venv
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install deps
pip install -r Interactive_Tool/requirements.txt

# 3. (Optional) Pre-train sklearn artifacts (~3-5 min)
python Interactive_Tool/case_analyzer.py

# 4. Launch the UI
python Interactive_Tool/app.py
```

Open <http://localhost:7860>. Pass `--share` to mint a temporary public URL
or `--force-retrain` to ignore the cached pipeline. NLTK corpora are
downloaded lazily on first analysis.

### Option C: Notebook (development + grading)

Open `Data_Exploration/data_exploration.ipynb` in JupyterLab and run top to
bottom. The notebook contains the entire 11-section pipeline including the
ROUGE evaluation, DistilBERT fine-tuning, and confusion matrices.

Additional notebook dependencies on top of `Interactive_Tool/requirements.txt`:

```bash
pip install rouge-score wordcloud matplotlib tensorflow ipywidgets python-dotenv tqdm
```

---

## Using the interactive tool

1. Pick an example from the **Load an example case** dropdown (or paste your
   own complaint / docket text).
2. Click **Analyze case**.
3. The right pane shows:
   - Raw / NLTK-cleaned / TF-IDF-excerpt token counts
   - The `class_action_sought` prediction with per-class confidence bars
   - The `case_type` prediction with per-class confidence bars
   - DistilBART **cascade** long / short / tiny abstractive summaries
   - Collapsible previews of the **TF-IDF extractive excerpt** (the BART input)
     and the **NLTK-cleaned text** (the classifier input)

---

## Re-training and updating models

- **From inside Docker:**
  ```bash
  docker compose -f Interactive_Tool/docker-compose.yml run --rm case-analyzer \
      python case_analyzer.py --force
  ```
- **Native:** `python Interactive_Tool/case_analyzer.py --force`

The retrained `sklearn_pipeline.joblib` is written to the `artifacts/` directory
(named volume `sklearn-artifacts` in the Docker path), so the next UI launch
loads it instantly.

---

## Findings summary

See `Data_Exploration/ANALYSIS.md` for a section-by-section writeup of the
notebook's results, including ROUGE scores per summary tier, LR / DistilBERT
test-set accuracies, the confusion matrix observations, and the LR-coefficient
inspection that surfaces *which terms* the production model keys on.
