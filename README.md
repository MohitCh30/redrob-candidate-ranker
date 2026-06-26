# Redrob Candidate Ranker

Submission for the **India Runs Data & AI Challenge** by Redrob x Hack2Skill.
Problem statement: Intelligent Candidate Discovery & Ranking.

**Live Demo:** https://huggingface.co/spaces/MohitML10/Intelligent_Candidate_Discovery

---

## What this does

Given a pool of 100,000 candidate profiles and a job description for a Senior AI Engineer role, this system ranks candidates the way a good recruiter would -- not by counting keywords, but by reading what candidates actually did.

It outputs a shortlist of the top 100 candidates with a score and a 1-2 sentence reasoning for each, generated directly from the candidate's profile fields with zero LLM involvement.

---

## How to run

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

Runs in 42 seconds on CPU. No GPU. No network calls. No pip dependencies for the ranker itself.

To validate the output:

```bash
python validate_submission.py submission.csv
```

To convert to XLSX for portal submission:

```bash
python -c "import pandas as pd; pd.read_csv('submission.csv').to_excel('submission.xlsx', index=False)"
```

---

## Architecture

```
candidates.jsonl.gz (100K profiles)
        |
        v
+-------------------------+
|       Pass 1            |
|  Role fit gate          |
|  Honeypot detection     |
|  Build IDF table        |
+-------------------------+
        |
        v
+-------------------------+     +-----------------+
|       Pass 2            | <-- |   IDF table     |
|  Skills score (0.30)    |     | term -> log IDF |
|  TF-IDF career (0.10)   |     +-----------------+
|  Trajectory  (0.25)     |
|  Education   (0.05)     |
|  x Behavioral multiplier|
+-------------------------+
        |
        v
+-------------------------+
|   Min-heap top-100      |
|   Normalize to [0,1]    |
|   Generate reasoning    |
+-------------------------+
        |
        v
   submission.csv
```

---

## Scoring components

**Skills match (0.30)**
Each skill claim is weighted by three evidence signals: proficiency level, endorsement count, and months of actual usage. A skill listed as "expert" with 0 endorsements and 0 months of usage scores near zero. A skill with "advanced" proficiency, 20+ endorsements, and 18 months of use scores full. This catches keyword stuffers without needing an LLM.

**Career TF-IDF (0.10)**
Corpus-wide TF-IDF is computed over all 100K career description texts. This measures how central retrieval and search work is to a candidate's actual career history, not just their skills list. A candidate who mentions "embedding", "FAISS", and "retrieval" repeatedly across multiple jobs scores high. A candidate who mentions it once in passing scores low.

**Career trajectory (0.25)**
Scores product-company tenure in relevant roles (months spent doing IR/search work at non-services firms, capped at 5 years), experience range fit (4-10 years is ideal), title relevance, and India location or willingness to relocate.

**Education (0.05)**
Light signal based on institution tier. Not a gating factor.

**Behavioral multiplier (0.4x to 1.2x)**
Applied on top of the raw score using Redrob platform signals:
- Open to work + active within 30 days: 1.15x
- GitHub activity score above 60: 1.1x
- Relevant assessment score above 75: 1.1x
- Recruiter response rate below 15%: 0.6x
- Interview completion rate below 40%: 0.7x
- Notice period above 90 days: 0.85x
- Inactive for 90+ days and not open to work: 0.5x

---

## Trap handling

The dataset contains keyword stuffers, services-only candidates, CV/speech domain specialists, and ~80 honeypot profiles with impossible data.

The role fit gate eliminates:
- Candidates whose entire career history is non-technical (HR, sales, marketing, finance)
- Candidates who have only worked at pure IT services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL) with no product company exposure
- Candidates whose primary domain is computer vision, speech, or robotics with no IR/NLP work in their descriptions

Honeypot detection flags:
- Any skill listed as "expert" proficiency with 0 duration months
- Any career entry where tenure exceeds the time since the company could have existed

Both are scored 0 and excluded from the top 100. The final submission contains 0 detected honeypots.

---

## Performance

| Metric | Value |
|---|---|
| Runtime | 42 seconds |
| Peak RAM | 1.66 GB |
| Unique scores in top 100 | 100 / 100 |
| Honeypots in top 100 | 0 |
| Validator | Passes clean |

---

## Files

```
rank.py                    # Main ranker, pure Python stdlib
app.py                     # Gradio sandbox for HuggingFace Spaces
validate_submission.py     # Official format validator
candidate_schema.json      # Schema reference for candidate profiles
sample_candidates.json     # First 50 candidates for testing
submission.csv             # Final ranked output (100 candidates)
submission.xlsx            # Same output in XLSX format for portal
submission_metadata.yaml   # Submission metadata
requirements.txt           # Dependencies for app.py only (gradio, pandas, openpyxl)
```

The full candidate dataset (candidates.jsonl.gz, 487 MB) is available via the challenge link and is not included in this repo.

---

## Dependencies

`rank.py` has zero external dependencies. It uses only Python stdlib: `re`, `math`, `heapq`, `gzip`, `csv`, `json`, `datetime`.

`app.py` (the Gradio sandbox) requires:

```
gradio
pandas
openpyxl
```

---

## Team

Team name: Residual Complex
Participant: Mohit Chaudhary
