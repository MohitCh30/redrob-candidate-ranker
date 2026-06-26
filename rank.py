#!/usr/bin/env python3
"""Rank Redrob candidates for the Senior AI Engineer search role."""

import argparse
import csv
import gzip
import heapq
import json
import re
from datetime import date
from math import log
from pathlib import Path
import pandas as pd



REFERENCE_DATE = date(2026, 6, 25)
SERVICES_FIRMS = {
    "accenture", "capgemini", "cognizant", "hcl", "infosys", "tcs", "wipro"
}
YOUNG_COMPANY_FOUNDING = {
    "krutrim": date(2023, 4, 1),
    "sarvam ai": date(2023, 7, 1),
}
NONTECH_RE = re.compile(
    r"\b(hr|human resources|sales|marketing|content|accountant|finance|"
    r"graphic designer|customer support|operations manager)\b",
    re.I,
)
TECH_RE = re.compile(
    r"\b(engineer|developer|data|software|machine learning|ml|ai|nlp|"
    r"search|ranking|analytics|scientist|devops|cloud|qa)\b",
    re.I,
)
RECENT_TITLE_RE = re.compile(
    r"\b(ml|machine learning|ai|data scientist|nlp|search|ranking|engineer)\b",
    re.I,
)
CV_DOMAIN_RE = re.compile(
    r"\b(computer vision|image classification|object detection|opencv|yolo|"
    r"speech recognition|speech|asr|tts|robotics)\b",
    re.I,
)
IR_CONTEXT_RE = re.compile(
    r"\b(embedding|semantic search|vector (?:database|search)|faiss|pinecone|"
    r"qdrant|weaviate|milvus|elasticsearch|opensearch|retrieval|rank(?:ing|er)?|"
    r"information retrieval|rag|recommendation system|ndcg|mrr|bm25)\b",
    re.I,
)

CORE_SKILLS = {
    "embeddings", "semantic search", "sentence transformers", "vector database",
    "vector search", "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "elasticsearch", "opensearch", "retrieval", "information retrieval",
    "information retrieval systems", "ranking", "ranking systems", "rag",
    "learning to rank", "search discovery", "search infrastructure",
}
SUPPORTING_SKILLS = {
    "python", "llm", "llms", "fine tuning", "fine tuning llms", "lora", "qlora",
    "peft", "nlp", "natural language processing", "ndcg", "evaluation",
    "a b testing", "recommendation systems", "recommendation system", "search",
    "bm25", "content matching", "indexing algorithms", "search backend",
}


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def months_between(start, end):
    return (
        (end.year - start.year) * 12
        + end.month
        - start.month
        - (end.day < start.day)
    )


def open_candidates(path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8")


def iter_candidates(path):
    with open_candidates(path) as handle:
        first = ""
        for line in handle:
            if line.strip():
                first = line
                break
        if not first:
            return
        if first.lstrip().startswith("["):
            with open_candidates(path) as array_handle:
                yield from json.load(array_handle)
            return
        yield json.loads(first)
        for line in handle:
            if line.strip():
                yield json.loads(line)


def evidence_multiplier(skill):
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)
    proficiency = skill.get("proficiency", "").lower()
    if endorsements > 10 and duration > 12 and proficiency in {"advanced", "expert"}:
        return 1.0
    if 1 <= endorsements <= 10 or 6 <= duration <= 12:
        return 0.6
    return 0.2


def is_honeypot(candidate):
    for skill in candidate.get("skills", []):
        if skill.get("proficiency", "").lower() == "expert" and skill.get("duration_months", 0) == 0:
            return True

    for job in candidate.get("career_history", []):
        duration = job.get("duration_months", 0)
        start = date.fromisoformat(job["start_date"])
        end = date.fromisoformat(job["end_date"]) if job.get("end_date") else REFERENCE_DATE
        if duration > months_between(start, end) + 1:
            return True
        founded = YOUNG_COMPANY_FOUNDING.get(job.get("company", "").lower())
        if founded and duration > months_between(founded, REFERENCE_DATE) + 1:
            return True
    return False


def role_fit(candidate):
    profile = candidate["profile"]
    history = candidate["career_history"]
    titles = [profile.get("current_title", "")] + [job.get("title", "") for job in history]

    def purely_nontechnical(title):
        return bool(NONTECH_RE.search(title)) and not TECH_RE.search(title)

    if purely_nontechnical(titles[0]) and all(purely_nontechnical(title) for title in titles[1:]):
        return False
    if history and all(job.get("company", "").lower() in SERVICES_FIRMS for job in history):
        return False

    work_text = " ".join(job.get("description", "") for job in history)
    domain_text = " ".join([
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        work_text,
    ])
    cv_is_primary = bool(CV_DOMAIN_RE.search(profile.get("current_title", "")))
    cv_is_primary = cv_is_primary or len(CV_DOMAIN_RE.findall(domain_text)) >= 3
    if cv_is_primary and not IR_CONTEXT_RE.search(work_text):
        return False
    return True


def score_skills(candidate):
    matches = []
    for skill in candidate.get("skills", []):
        name = normalized(skill.get("name", ""))
        if name in CORE_SKILLS:
            matches.append((1.0 * evidence_multiplier(skill), 1, skill))
        elif name in SUPPORTING_SKILLS:
            matches.append((0.55 * evidence_multiplier(skill), 0, skill))

    work_text = " ".join(job.get("description", "") for job in candidate["career_history"])
    context_terms = {normalized(match.group(0)) for match in IR_CONTEXT_RE.finditer(work_text)}
    score = min(1.0, sum(item[0] for item in matches) / 10.0)
    strongest = max(matches, default=(0.0, 0, None), key=lambda item: (item[0], item[1]))
    return score, strongest[2], len(context_terms)


def tfidf_career_score(candidate, idf_table):
    text = " ".join(job.get("description", "") for job in candidate["career_history"])
    word_count = len(re.findall(r"\b\w+\b", text))
    if not word_count:
        return 0.0
    term_counts = {}
    for match in IR_CONTEXT_RE.finditer(text):
        term = normalized(match.group(0))
        term_counts[term] = term_counts.get(term, 0) + 1
    return sum(count / word_count * idf_table[term] for term, count in term_counts.items())


def is_product_job(job):
    return (
        job.get("company", "").lower() not in SERVICES_FIRMS
        and job.get("industry", "").lower() not in {"it services", "consulting", "ai services"}
    )


def score_career(candidate):
    profile = candidate["profile"]
    history = candidate["career_history"]
    years = profile.get("years_of_experience", 0)
    if 4 <= years <= 10:
        experience = 1.0
    elif 3 <= years < 4 or 10 < years <= 14:
        experience = 0.7
    else:
        experience = 0.3

    product_exposure = any(is_product_job(job) for job in history)
    relevant_months = sum(
        job.get("duration_months", 0)
        for job in history
        if is_product_job(job) and IR_CONTEXT_RE.search(job.get("description", ""))
    )
    tenure_score = min(5.0, relevant_months / 12.0) / 5.0
    recent_title = bool(RECENT_TITLE_RE.search(profile.get("current_title", "")))
    no_nontech_title = not any(
        NONTECH_RE.search(job.get("title", "")) and not TECH_RE.search(job.get("title", ""))
        for job in history
    )
    location_fit = (
        profile.get("country", "").lower() == "india"
        or candidate["redrob_signals"].get("willing_to_relocate", False)
    )
    total = (
        experience
        + 0.4 * tenure_score
        + 0.3 * recent_title
        + 0.2 * no_nontech_title
        + 0.1 * location_fit
    )
    return total / 2.0, product_exposure, recent_title, tenure_score


def score_behavior(candidate):
    signals = candidate["redrob_signals"]
    last_active = date.fromisoformat(signals["last_active_date"])
    inactive_days = (REFERENCE_DATE - last_active).days
    multiplier = 1.0
    if not signals.get("open_to_work_flag", False) and inactive_days > 90:
        multiplier *= 0.5
    if signals.get("recruiter_response_rate", 0) < 0.15:
        multiplier *= 0.6
    if signals.get("interview_completion_rate", 0) < 0.4:
        multiplier *= 0.7
    if signals.get("notice_period_days", 0) > 90:
        multiplier *= 0.85
    if signals.get("github_activity_score", -1) > 60:
        multiplier *= 1.1

    relevant_assessment = max(
        (
            score
            for skill, score in signals.get("skill_assessment_scores", {}).items()
            if normalized(skill) in CORE_SKILLS | SUPPORTING_SKILLS
        ),
        default=-1,
    )
    if relevant_assessment > 75:
        multiplier *= 1.1
    if signals.get("open_to_work_flag", False) and inactive_days <= 30:
        multiplier *= 1.15
    return min(1.2, max(0.4, multiplier)), inactive_days, relevant_assessment


def score_education(candidate):
    values = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4, "tier_4": 0.4, "unknown": 0.5}
    return max((values.get(item.get("tier", "unknown"), 0.5) for item in candidate["education"]), default=0.5)


def score_candidate(candidate, tfidf_score=0.0):
    honeypot = is_honeypot(candidate)
    if not honeypot and not role_fit(candidate):
        return None

    skills, strongest, context_count = score_skills(candidate)
    career, product_exposure, recent_title, tenure_score = score_career(candidate)
    education = score_education(candidate)
    behavior, inactive_days, assessment = score_behavior(candidate)
    raw = 0.0 if honeypot else (
        skills * 0.30
        + career * 0.25
        + tfidf_score * 0.10
        + education * 0.05
    ) * behavior
    return {
        "candidate_id": candidate["candidate_id"],
        "raw": raw,
        "profile": candidate["profile"],
        "signals": candidate["redrob_signals"],
        "strongest": strongest,
        "context_count": context_count,
        "product_exposure": product_exposure,
        "tenure_score": tenure_score,
        "tfidf_score": tfidf_score,
        "recent_title": recent_title,
        "inactive_days": inactive_days,
        "assessment": assessment,
        "honeypot": honeypot,
    }


def evidence_text(skill):
    if not skill:
        return "no direct retrieval/search skill in the skills list"
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)
    quality = evidence_multiplier(skill)
    label = "strong" if quality == 1.0 else "moderate" if quality == 0.6 else "light"
    return (
        f"{label} {skill['name']} evidence "
        f"({skill.get('proficiency', 'unknown')}, {endorsements} endorsements, {duration}mo)"
    )


def behavioral_text(item):
    signals = item["signals"]
    if signals.get("open_to_work_flag") and item["inactive_days"] <= 30:
        return "They are open to work and active within 30 days."
    if item["assessment"] > 75:
        return f"A relevant assessment score reaches {item['assessment']:.0f}."
    if signals.get("github_activity_score", -1) > 60:
        return f"GitHub activity is strong at {signals['github_activity_score']:.0f}/100."
    if signals.get("recruiter_response_rate", 0) < 0.15:
        return f"Recruiter response rate is only {signals['recruiter_response_rate']:.0%}."
    return f"Recruiter response rate is {signals.get('recruiter_response_rate', 0):.0%}."


def concern_text(item):
    profile, signals = item["profile"], item["signals"]
    if item["honeypot"]:
        return "Profile contains an impossible tenure or zero-duration expert claim."
    if signals.get("notice_period_days", 0) > 90:
        return f"Concern: {signals['notice_period_days']}-day notice period."
    if item["inactive_days"] > 90:
        return f"Concern: last active {item['inactive_days']} days ago."
    if signals.get("recruiter_response_rate", 0) < 0.15:
        return "Concern: low recruiter responsiveness."
    if profile.get("country", "").lower() != "india" and not signals.get("willing_to_relocate"):
        return "Concern: outside India and not willing to relocate."
    if not item["product_exposure"]:
        return "Concern: no clear product-company exposure."
    if item["context_count"] == 0:
        return "Concern: retrieval evidence is not repeated in work descriptions."
    if not item["recent_title"]:
        return "Concern: current title is adjacent rather than directly ML/search-focused."
    return "No major availability concern is visible."


def reasoning(item):
    profile = item["profile"]
    title = profile.get("current_title", "Candidate")
    years = profile.get("years_of_experience", 0)
    evidence = evidence_text(item["strongest"])
    behavior = behavioral_text(item)
    concern = concern_text(item)
    variant = int(item["candidate_id"][-2:]) % 4
    if variant == 0:
        return f"{title} with {years:g} years; {evidence}, backed by {item['context_count']} work-context retrieval signal(s). {behavior} {concern}"
    if variant == 1:
        return f"At {years:g} years' experience, this {title} shows {evidence}. {behavior} {concern}"
    if variant == 2:
        exposure = "product-company experience" if item["product_exposure"] else "services-heavy experience"
        return f"{title}, {years:g} years, with {exposure} and {evidence}. {behavior} {concern}"
    return f"{evidence.capitalize()} supports this {title}'s {years:g}-year trajectory. {behavior} {concern}"


def rank_candidates(path, limit):
    candidates = list(iter_candidates(path))
    document_frequency = {}
    for candidate in candidates:
        text = " ".join(job.get("description", "") for job in candidate["career_history"])
        for term in {normalized(match.group(0)) for match in IR_CONTEXT_RE.finditer(text)}:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    total_candidates = len(candidates)
    idf_table = {
        term: log(total_candidates / (1 + frequency))
        for term, frequency in document_frequency.items()
    }

    tfidf_scores = [tfidf_career_score(candidate, idf_table) for candidate in candidates]
    max_tfidf = max(tfidf_scores, default=0.0)

    heap = []
    minimum = float("inf")
    maximum = float("-inf")
    for candidate, tfidf_score in zip(candidates, tfidf_scores):
        normalized_tfidf = tfidf_score / max_tfidf if max_tfidf else 0.0
        item = score_candidate(candidate, normalized_tfidf)
        if item is None:
            continue
        raw = item["raw"]
        minimum = min(minimum, raw)
        maximum = max(maximum, raw)
        candidate_number = int(item["candidate_id"][5:])
        entry = (raw, -candidate_number, item)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    if len(heap) < limit:
        raise ValueError(f"Only {len(heap)} eligible candidates; cannot write {limit} rows")
    spread = maximum - minimum
    ranked = []
    for _, _, item in heap:
        score = 1.0 if spread == 0 else (item["raw"] - minimum) / spread
        item["score"] = round(score, 9)
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-item["score"], item["candidate_id"]))


def write_submission(path, ranked):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(ranked, 1):
            writer.writerow([item["candidate_id"], rank, f"{item['score']:.9f}", reasoning(item)])


def self_check():
    assert evidence_multiplier({"endorsements": 11, "duration_months": 13, "proficiency": "expert"}) == 1.0
    assert evidence_multiplier({"endorsements": 4, "duration_months": 2, "proficiency": "beginner"}) == 0.6
    assert months_between(date(2025, 1, 1), date(2026, 1, 1)) == 12
    candidate = {"career_history": [{"description": "FAISS retrieval retrieval", "duration_months": 12}]}
    assert abs(tfidf_career_score(candidate, {"faiss": 2.0, "retrieval": 1.0}) - 4 / 3) < 1e-12
    career_candidate = {
        "profile": {"years_of_experience": 6, "current_title": "Search Engineer", "country": "India"},
        "career_history": [{
            "company": "Swiggy", "industry": "Software", "title": "Search Engineer",
            "description": "Built retrieval systems", "duration_months": 36,
        }],
        "redrob_signals": {"willing_to_relocate": False},
    }
    assert score_career(career_candidate)[3] == 0.6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    self_check()
    write_submission(args.out, rank_candidates(args.candidates, args.limit))


if __name__ == "__main__":
    main()
