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


def result_to_markdown(result: dict) -> str:
    if err := result.get("error"):
        return err
    s = result["summaries"]
    return f"""### Summaries

**Long**  
{s["long"]}

**Short**  
{s["short"]}

**Tiny**  
{s["tiny"]}

---

### Predictions

| Field | Prediction |
|-------|------------|
| Class action sought | `{result["class_action_sought"]}` |
| Case type | `{result["case_type"]}` |
"""


def build_ui(analyzer: CaseAnalyzer) -> gr.Blocks:
    def analyze(text: str) -> str:
        return result_to_markdown(analyzer.analyze(text or ""))

    with gr.Blocks(title="Multi-LexSum Case Analyzer") as demo:
        gr.Markdown(
            """
# Interactive case analyzer

Paste legal case text (for example concatenated docket-style sources). The tool returns **DistilBART**
long / short / tiny summaries and **logistic regression** predictions for **class action sought** and **case type**,
using the same preprocessing and training recipe as `Data_Exploration/exploration_paul.ipynb` (Task 7).

The first summary run downloads the summarization model; expect on the order of tens of seconds per analysis on CPU.
"""
        )
        inp = gr.Textbox(
            label="Case text",
            placeholder="Paste a legal case description here…",
            lines=14,
        )
        btn = gr.Button("Analyze case", variant="primary")
        out = gr.Markdown()
        btn.click(fn=analyze, inputs=inp, outputs=out, show_progress="full")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached TF-IDF / LR artifacts and retrain from the dataset.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio public link.")
    args = parser.parse_args()

    status: list[str] = []

    def log(msg: str) -> None:
        status.append(msg)
        print(msg)

    log("Initializing models (sklearn cache + optional BART download)…")
    analyzer = CaseAnalyzer()
    analyzer.ensure_ready(log=log, force_retrain=args.force_retrain)
    log("Sklearn pipeline ready. Launching UI…")

    demo = build_ui(analyzer)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
