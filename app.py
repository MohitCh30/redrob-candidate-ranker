"""
Redrob Hackathon — Candidate Ranker Sandbox
Upload a JSON file of candidates and get back a ranked shortlist.
Accepts: JSON array (like sample_candidates.json) or JSONL (one candidate per line).
"""

import json
import tempfile
import gradio as gr
import pandas as pd

from rank import rank_candidates, reasoning


def run_ranker(file, top_n):
    if file is None:
        return None, "Upload a candidate JSON file to get started."

    try:
        top_n = int(top_n)
    except (ValueError, TypeError):
        top_n = 10

    top_n = max(1, min(top_n, 100))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="wb") as tmp:
        tmp.write(file)
        tmp_path = tmp.name

    try:
        ranked = rank_candidates(tmp_path, limit=top_n)
    except ValueError as e:
        return None, f"Ranking error: {e}"

    rows = []
    for rank_pos, item in enumerate(ranked, 1):
        rows.append({
            "Rank": rank_pos,
            "Candidate ID": item["candidate_id"],
            "Score": round(item["score"], 4),
            "Reasoning": reasoning(item),
        })

    df = pd.DataFrame(rows)
    return df, f"Ranked {len(rows)} candidates successfully."


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("## Redrob Hackathon — Candidate Ranker")
    gr.Markdown(
        "Upload a JSON file (array format like `sample_candidates.json`, or JSONL) "
        "and get back a ranked shortlist for the Senior AI Engineer role."
    )

    with gr.Row():
        file_input = gr.File(label="Candidate JSON / JSONL", type="binary")
        top_n_input = gr.Number(label="Top N (max 100)", value=10, precision=0)

    run_btn = gr.Button("Run Ranker")
    status = gr.Textbox(label="Status", interactive=False)
    output_table = gr.Dataframe(label="Ranked Candidates", wrap=True)

    run_btn.click(
        fn=run_ranker,
        inputs=[file_input, top_n_input],
        outputs=[output_table, status],
    )

if __name__ == "__main__":
    demo.launch()
