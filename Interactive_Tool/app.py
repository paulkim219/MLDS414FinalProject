"""
Standalone web UI for the Task 7 interactive case analyzer (Gradio).

Run from repo root or this folder:
  pip install -r Interactive_Tool/requirements.txt
  python Interactive_Tool/app.py
"""

from __future__ import annotations

import argparse

import gradio as gr

from case_analyzer import CaseAnalyzer

EXAMPLE_CASES: dict[str, str] = {
    "Employment discrimination (EEOC v. employer)": (
        "On September 15, 2005, the Equal Employment Opportunity Commission filed suit in the "
        "United States District Court for the Southern District of Alabama against House of "
        "Philadelphia Center, Inc., on behalf of a former employee who was allegedly fired because "
        "she was pregnant. The complaint, brought under Title VII of the Civil Rights Act of 1964 "
        "and Title I of the Civil Rights Act of 1991, alleged unlawful employment practices on the "
        "basis of sex (pregnancy). The plaintiff sought injunctive relief, back pay, compensatory "
        "and punitive damages. The defendant operated a non-profit shelter serving victims of "
        "domestic violence. After a period of discovery, the parties entered into a consent decree "
        "providing monetary relief and mandatory anti-discrimination training for all supervisory "
        "personnel."
    ),
    "Immigration / border detention (class action)": (
        "Plaintiffs, a putative class of immigrant detainees held at the South Texas Family "
        "Residential Center, filed this action in the United States District Court for the Western "
        "District of Texas on behalf of themselves and all others similarly situated. The complaint "
        "alleges that the Department of Homeland Security and U.S. Immigration and Customs "
        "Enforcement maintain a policy of prolonged detention of asylum seekers in violation of the "
        "Due Process Clause of the Fifth Amendment, the Immigration and Nationality Act, and the "
        "Administrative Procedure Act. Plaintiffs seek declaratory and injunctive relief, including "
        "a court order requiring individualized custody determinations and access to counsel. "
        "Plaintiffs also seek class certification under Federal Rule of Civil Procedure 23(b)(2)."
    ),
    "Prison conditions / Eighth Amendment": (
        "This is a 1983 civil rights action filed by inmates in the custody of the State Department "
        "of Corrections, on behalf of themselves and a class of similarly situated prisoners. The "
        "complaint alleges systemic deliberate indifference to serious medical needs, chronic "
        "overcrowding exceeding 180% of design capacity, and a pattern of excessive force by "
        "correctional officers, in violation of the Eighth Amendment's prohibition on cruel and "
        "unusual punishment. Plaintiffs seek a declaratory judgment, prospective injunctive relief "
        "including population caps and independent medical oversight, and certification of a "
        "plaintiff class under Rule 23(b)(2). The action follows a multi-year administrative "
        "grievance process and prior reports by a court-appointed special master documenting "
        "ongoing constitutional violations across all facilities operated by the defendants."
    ),
}


def _format_probs(probs: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(((label, p) for label, p in probs.items()), key=lambda x: x[1], reverse=True)


def _prob_bar_markdown(probs: dict[str, float]) -> str:
    rows = []
    for label, p in _format_probs(probs):
        filled = round(p * 20)
        bar = "█" * filled + "░" * (20 - filled)
        rows.append(f"`{bar}` **{p*100:5.1f}%**  {label}")
    return "\n\n".join(rows)


def result_to_markdown(result: dict) -> str:
    if err := result.get("error"):
        return f"### Error\n\n{err}"

    s = result["summaries"]
    action_bars = _prob_bar_markdown(result["class_action_probs"])
    type_bars = _prob_bar_markdown(result["case_type_probs"])

    return f"""### Input statistics

- Raw characters: **{result['raw_char_count']:,}**
- NLTK-cleaned tokens: **{result['clean_word_count']:,}**
- TF-IDF excerpt tokens (BART input): **{result['excerpt_word_count']:,}**

### Predictions

**Class action sought:** `{result['class_action_sought']}`

{action_bars}

**Case type:** `{result['case_type']}`

{type_bars}

---

### Summaries (DistilBART cascade — excerpt → long → short → tiny)

**Long**
{s["long"]}

**Short**
{s["short"]}

**Tiny**
{s["tiny"]}

---

<details>
<summary>TF-IDF extractive excerpt (BART input — first 500 chars)</summary>

```
{result['excerpt_preview']}
```
</details>

<details>
<summary>NLTK-cleaned text (classifier input — first 500 chars)</summary>

```
{result['cleaned_preview']}
```
</details>
"""


def build_ui(analyzer: CaseAnalyzer) -> gr.Blocks:
    def analyze(text: str) -> str:
        return result_to_markdown(analyzer.analyze(text or ""))

    def load_example(name: str) -> str:
        return EXAMPLE_CASES.get(name, "")

    with gr.Blocks(title="Multi-LexSum Case Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Multi-LexSum Interactive Case Analyzer

Paste legal case text (for example concatenated complaint / docket text). The pipeline:

1. **Legal-boilerplate stripping** — regex removal of dockets, U.S.C. citations, reporter citations, court headers.
2. **NLTK normalization** — `word_tokenize` → stopwords → WordNet lemmatization, feeding the classifiers.
3. **TF-IDF extractive summarization** — sentence-level TF-IDF ranking selects a 480-token excerpt from anywhere in the document.
4. **DistilBART cascade summarization** — excerpt → long → short → tiny, mirroring how the dataset's experts wrote ground truth.
5. **Logistic-regression classifiers** for `class_action_sought` and `case_type`, with per-class confidence bars.

Uses the same preprocessing and training recipe as `Data_Exploration/data_exploration.ipynb`.
The first summary run downloads the BART model (~1.2 GB); expect tens of seconds per analysis on CPU.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                example_dropdown = gr.Dropdown(
                    label="Load an example case",
                    choices=[""] + list(EXAMPLE_CASES.keys()),
                    value="",
                )
                inp = gr.Textbox(
                    label="Case text",
                    placeholder="Paste a legal case description here...",
                    lines=18,
                )
                with gr.Row():
                    btn = gr.Button("Analyze case", variant="primary")
                    clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                out = gr.Markdown(label="Results")

        example_dropdown.change(fn=load_example, inputs=example_dropdown, outputs=inp)
        btn.click(fn=analyze, inputs=inp, outputs=out, show_progress="full")
        clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[inp, out])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached TF-IDF / LR artifacts and retrain from the dataset.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio public link.")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg)

    log("Initializing models (sklearn cache + optional BART download)...")
    analyzer = CaseAnalyzer()
    analyzer.ensure_ready(log=log, force_retrain=args.force_retrain)
    log("Sklearn pipeline ready. Launching UI...")

    demo = build_ui(analyzer)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
