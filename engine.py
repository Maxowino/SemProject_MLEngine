"""
engine.py - KNN + RAG logic for the Flask API.

Ported from KNN_model.ipynb and RAG_feedback.ipynb so the API does not depend on the notebooks.
this module only READS data/expanded_software_questions.csv,
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# Settings

DATA_FILE = Path(os.getenv("QUESTIONS_CSV",
                           Path(__file__).resolve().parent / "data" / "expanded_software_questions.csv"))
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
STRONG, PASS = 80, 50          # score thresholds
TOP_RELATED = 3                # related study questions per weak answer (RAG)
MIN_SIMILARITY = 0.05          # RAG: ignore questions that share almost nothing
MAX_ANSWER_CHARS = 2000
LLM_MODEL = "claude-sonnet-5-5"  # only used if ANTHROPIC_API_KEY is set

CAREER_MAP = {
    "Software Engineering": [
        "General Programming", "General Program", "Programming Fundamentals",
        "Object-Oriented Programming", "Software Engineering", "Software Testing",
        "Version Control", "Software Development Tools", "Languages and Frameworks",
        "Data Structures", "Algorithms", "Data Structures and Algorithms",
    ],
    "Web Development": ["Front-end", "Back-end", "Full-stack", "Web Development", "Networking and Web"],
    "Systems and DevOps": ["DevOps", "System Design", "Distributed Systems", "Networking",
                           "Low-level Systems", "Operating Systems"],
    "Data and AI": ["Database and SQL", "Databases", "Database Systems", "Data Engineering",
                    "Machine Learning", "Artificial Intelligence"],
    "Cybersecurity": ["Security"],
}
_CATEGORY_TO_CAREER = {cat: career for career, cats in CAREER_MAP.items() for cat in cats}


def _normalize(text):
    return " ".join(str(text).casefold().split())



# Dataset + models (built once, at import)

def _load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, encoding="cp1252")

    required = ["Question", "Answer", "Category", "Difficulty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["Question"] != "") & (df["Answer"] != "") & (df["Category"] != "") & (df["Difficulty"] != "")]

    df = df.assign(_norm=df["Question"].map(_normalize))
    df = df.drop_duplicates(subset="_norm").drop(columns="_norm").reset_index(drop=True)

    df["Category"] = df["Category"].replace({"General Program": "General Programming"})
    df["Career"] = df["Category"].map(_CATEGORY_TO_CAREER)
    unmapped = df.loc[df["Career"].isna(), "Category"].unique()
    if len(unmapped):
        raise ValueError(f"Add these categories to CAREER_MAP: {list(unmapped)}")
    return df


knn_df = _load_dataset()
knn_df["SearchText"] = knn_df["Question"] + " " + knn_df["Category"]

# KNN model (next-question recommender)
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(knn_df["SearchText"])
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(5, len(knn_df)))
knn_model.fit(X)

# Answer scoring model
answer_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
answer_vectorizer.fit(knn_df["Answer"].astype(str).tolist())

# RAG knowledge base (question + category + answer)
kb_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
kb_matrix = kb_vectorizer.fit_transform(knn_df["Question"] + " " + knn_df["Category"] + " " + knn_df["Answer"])

_QUESTION_INDEX = {_normalize(q): i for i, q in enumerate(knn_df["Question"])}


# --------------------------------------------------------------------------------------
# KNN: recommend, score, difficulty
# --------------------------------------------------------------------------------------
def recommend_similar_questions(question, top_n=3, difficulty=None, category=None, career=None):
    if not str(question).strip():
        raise ValueError("Enter a question or topic.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    query_vector = vectorizer.transform([str(question).strip()])
    distances, indices = knn_model.kneighbors(query_vector, n_neighbors=len(knn_df))

    results = knn_df.iloc[indices[0]].copy()
    results["Similarity"] = 1 - distances[0]
    results = results[results["Question"].map(_normalize) != _normalize(question)]

    if difficulty is not None:
        results = results[results["Difficulty"].str.casefold() == difficulty.casefold()]
    if category is not None:
        results = results[results["Category"].str.casefold() == category.casefold()]
    if career is not None:
        results = results[results["Career"].str.casefold() == career.casefold()]

    results = results.sort_values("Similarity", ascending=False).head(top_n)
    return results[["Question", "Answer", "Category", "Career", "Difficulty", "Similarity"]].reset_index(drop=True)


def _keywords(text):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b", str(text).lower())
    return {w for w in words if len(w) > 2 and w not in ENGLISH_STOP_WORDS}


def score_answer(question, user_answer):
    """Score a user's answer against the dataset's expected answer (0-100)."""
    idx = _QUESTION_INDEX.get(_normalize(question))
    if idx is None:
        raise ValueError("Question not found in the dataset.")
    expected = knn_df.iloc[idx]["Answer"]

    if not str(user_answer).strip():
        return {"score": 0, "similarity": 0, "keyword_coverage": 0, "matched_keywords": [],
                "missing_keywords": [], "expected_answer": expected,
                "feedback": "No answer provided. Review the expected answer and try again."}

    pair = answer_vectorizer.transform([expected, str(user_answer)])
    similarity = float(cosine_similarity(pair[0:1], pair[1:2])[0][0])

    exp_kw, usr_kw = _keywords(expected), _keywords(user_answer)
    if exp_kw:
        matched, missing = exp_kw & usr_kw, exp_kw - usr_kw
        coverage = len(matched) / len(exp_kw)
    else:
        matched, missing, coverage = set(), set(), 0.0

    score = round((0.60 * similarity + 0.40 * coverage) * 100)

    if score >= STRONG:
        feedback = "Good answer. Review the missing concepts, if any, before moving on."
    elif score >= PASS:
        feedback = "Partially correct. Review the expected answer and strengthen the missing concepts."
    else:
        feedback = "Your answer needs improvement. Study the expected answer and try again."

    return {"score": score, "similarity": round(similarity * 100, 2), "keyword_coverage": round(coverage * 100, 2),
            "matched_keywords": sorted(matched), "missing_keywords": sorted(missing),
            "expected_answer": expected, "feedback": feedback}


def choose_next_difficulty(current_difficulty, recent_scores):
    """Average of the last 3 scores: < 50 goes down, >= 80 goes up, otherwise stays."""
    if current_difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"Difficulty must be one of {DIFFICULTY_LEVELS}")
    if not recent_scores:
        return current_difficulty
    avg = sum(recent_scores[-3:]) / len(recent_scores[-3:])
    i = DIFFICULTY_LEVELS.index(current_difficulty)
    if avg < PASS:
        i = max(0, i - 1)
    elif avg >= STRONG:
        i = min(len(DIFFICULTY_LEVELS) - 1, i + 1)
    return DIFFICULTY_LEVELS[i]


# --------------------------------------------------------------------------------------
# Interview session (adaptive KNN loop)
# --------------------------------------------------------------------------------------
class InterviewSession:
    def __init__(self, career=None, max_questions=10, starting_difficulty="Easy"):
        self.career = career
        self.max_questions = max_questions
        self.starting_difficulty = starting_difficulty
        self.asked_questions = []
        self.scores = []
        self.log = []
        self.current_difficulty = starting_difficulty
        self.finished = False
        self.last_knn = None

    def can_continue(self):
        return not self.finished and len(self.asked_questions) < self.max_questions

    def _levels_by_distance(self):
        i = DIFFICULTY_LEVELS.index(self.current_difficulty)
        prefer_lower = bool(self.scores) and self.scores[-1] < PASS
        return sorted(DIFFICULTY_LEVELS, key=lambda lvl: (
            abs(DIFFICULTY_LEVELS.index(lvl) - i),
            (DIFFICULTY_LEVELS.index(lvl) > i) if prefer_lower else (DIFFICULTY_LEVELS.index(lvl) < i)))

    def first_question(self, seed=None):
        import random
        rng = random.Random(seed)
        for level in self._levels_by_distance():
            pool = knn_df[knn_df["Difficulty"].str.casefold() == level.casefold()]
            if self.career is not None:
                pool = pool[pool["Career"] == self.career]
            if not pool.empty:
                return pool.iloc[rng.randrange(len(pool))]
        self.finished = True
        return None

    def record_answer(self, question, score, user_answer="", expected_answer="", difficulty=None):
        if not self.can_continue():
            self.finished = True
            return
        if question not in self.asked_questions:
            row = knn_df[knn_df["Question"] == question]
            if difficulty is None:
                difficulty = row.iloc[0]["Difficulty"] if not row.empty else self.current_difficulty
            self.asked_questions.append(question)
            self.scores.append(float(score))
            self.log.append({
                "question": question,
                "category": row.iloc[0]["Category"] if not row.empty else None,
                "difficulty": difficulty,
                "user_answer": user_answer,
                "expected_answer": expected_answer,
                "score": float(score),
            })
        self.current_difficulty = choose_next_difficulty(self.current_difficulty, self.scores[-3:])
        if len(self.asked_questions) >= self.max_questions:
            self.finished = True

    def next_question(self, topic, category=None):
        if not self.can_continue():
            self.finished = True
            return None
        for level in self._levels_by_distance():
            recs = recommend_similar_questions(topic, top_n=len(knn_df), difficulty=level,
                                               category=category, career=self.career)
            recs = recs[~recs["Question"].isin(self.asked_questions)]
            if not recs.empty:
                self.last_knn = {"topic": topic, "target": self.current_difficulty, "level": level,
                                 "candidates": recs.head(5).reset_index(drop=True)}
                return recs.iloc[0]
        self.finished = True
        return None

    def final_summary(self):
        total = len(self.scores)
        by_category = {}
        for e in self.log:
            by_category.setdefault(e["category"], []).append(e["score"])
        return {
            "career": self.career,
            "finished": self.finished,
            "questions_answered": total,
            "average_score": round(sum(self.scores) / total, 2) if total else 0,
            "scores": self.scores,
            "questions": self.asked_questions,
            "average_by_category": {c: round(sum(v) / len(v), 2) for c, v in by_category.items()},
            "weak_answers": [e for e in self.log if e["score"] < PASS],
            "log": self.log,
        }


# --------------------------------------------------------------------------------------
# RAG feedback (retrieval from the dataset + rule-based or optional LLM summary)
# --------------------------------------------------------------------------------------
_EXTRA_STOP = {"allows", "using", "used", "also", "like", "often", "such", "based", "called",
               "between", "within", "typically", "usually", "different", "various"}


def _key_terms(text):
    seen, terms = set(), []
    for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", str(text).lower()):
        if w in ENGLISH_STOP_WORDS or w in _EXTRA_STOP or w in seen:
            continue
        seen.add(w)
        terms.append(w)
    return terms


def _missing_terms(expected, user_answer, limit=8):
    user_stems = {w[:5] for w in _key_terms(user_answer)}
    return [w for w in _key_terms(expected) if w[:5] not in user_stems][:limit]


def _verdict(score):
    return "Strong" if score >= STRONG else "Partly correct" if score >= PASS else "Needs work"


def _performance_level(avg):
    return "Strong" if avg >= 70 else "Developing" if avg >= PASS else "Needs work"


def get_expected_answer(question):
    idx = _QUESTION_INDEX.get(_normalize(question))
    return knn_df.iloc[idx]["Answer"] if idx is not None else None


def retrieve_related(question, career, k=TOP_RELATED):
    idx = _QUESTION_INDEX.get(_normalize(question))
    category = knn_df.iloc[idx]["Category"] if idx is not None else None
    expected = knn_df.iloc[idx]["Answer"] if idx is not None else ""

    sims = cosine_similarity(kb_vectorizer.transform([question + " " + expected]), kb_matrix)[0]
    cand = knn_df.assign(similarity=sims)
    cand = cand[(cand["Question"] != question) & (cand["Career"] == career)]
    cand = cand.assign(same_category=(cand["Category"] == category))
    cand = cand[cand["same_category"] | (cand["similarity"] >= MIN_SIMILARITY)]
    cand = cand.sort_values(["same_category", "similarity"], ascending=False).head(k)
    return [{"question": r["Question"], "category": r["Category"], "difficulty": r["Difficulty"],
             "similarity": round(float(r["similarity"]), 3)} for _, r in cand.iterrows()]


def question_feedback(entry, career):
    expected = entry.get("expected_answer") or get_expected_answer(entry["question"]) or ""
    user_answer = entry.get("user_answer", "")
    score = entry["score"]
    v = _verdict(score)
    weak = score < STRONG

    if not str(user_answer).strip():
        text = "No answer was given. Read the model answer and try again."
    elif v == "Strong":
        text = "Strong answer. It covers the main points of the model answer."
    elif v == "Partly correct":
        text = "Partly correct. Add the missing key terms and compare your answer with the model answer."
    else:
        text = "This answer needs work. Study the model answer and the related questions."

    return {"question": entry["question"], "category": entry.get("category"),
            "difficulty": entry.get("difficulty"), "score": score, "verdict": v,
            "your_answer": user_answer, "model_answer": expected,
            "missing_terms": _missing_terms(expected, user_answer) if weak else [],
            "feedback": text,
            "study_next": retrieve_related(entry["question"], career) if weak else []}


def _rule_based_narrative(r):
    if r["questions_answered"] == 0:
        return "No questions were answered, so there is no feedback yet."
    parts = [f"You answered {r['questions_answered']} {r['career']} questions with an average score of "
             f"{r['average_score']:g}% ({r['performance_level']})."]
    if r["hardest_level_passed"]:
        parts.append(f"The hardest level you answered well was {r['hardest_level_passed']}.")
    if r["strong_topics"]:
        parts.append("Strong topics: " + ", ".join(r["strong_topics"]) + ".")
    if r["weak_topics"]:
        parts.append("Revise these first: " + ", ".join(w["category"] for w in r["weak_topics"][:3]) + ".")
    else:
        parts.append("No topic fell below the pass mark. Try another interview to reach harder questions.")
    return " ".join(parts)


def _llm_narrative(r):
    """Optional: only runs if ANTHROPIC_API_KEY is set. Falls back to the offline text on any error."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import json
    context = {"career": r["career"], "average_score": r["average_score"], "level": r["performance_level"],
               "strong_topics": r["strong_topics"], "weak_topics": r["weak_topics"],
               "answers": [{"question": q["question"], "score": q["score"], "missing_terms": q["missing_terms"],
                            "model_answer": q["model_answer"],
                            "related_questions": [s["question"] for s in q["study_next"]]}
                           for q in r["questions"]]}
    prompt = ("You are an interview coach for IT students. Write feedback for the candidate using ONLY the "
              "CONTEXT below. Do not invent facts or questions. Format: a 3-4 sentence overall assessment, then "
              "'Top priorities:' with 3 short bullet points. Address the candidate as 'you'.\n\nCONTEXT:\n"
              + json.dumps(context, indent=1))
    try:
        import anthropic
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=LLM_MODEL, max_tokens=700, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except Exception:
        return None


def build_report(summary):
    log = summary.get("log", [])
    career = summary.get("career")
    by_cat = summary.get("average_by_category", {})
    questions = [question_feedback(e, career) for e in log]

    weak_topics = sorted([{"category": c, "average_score": s} for c, s in by_cat.items() if s < PASS],
                         key=lambda w: w["average_score"])
    strong_topics = [c for c, s in by_cat.items() if s >= STRONG]
    passed = [q["difficulty"] for q in questions if q["score"] >= PASS and q["difficulty"] in DIFFICULTY_LEVELS]
    hardest = max(passed, key=DIFFICULTY_LEVELS.index) if passed else None

    plan = []
    for w in weak_topics:
        related = []
        for q in questions:
            if q["category"] == w["category"]:
                related += [s["question"] for s in q["study_next"] if s["question"] not in related]
        line = f"Revise {w['category']} (average {w['average_score']:g}%)."
        if related:
            line += " Practise: " + "; ".join(related[:3])
        plan.append(line)
    if not plan and questions:
        plan.append("Keep practising and aim for Hard questions in your next interview.")

    avg = summary.get("average_score", 0)
    report = {"career": career, "questions_answered": len(log), "average_score": avg,
              "performance_level": _performance_level(avg), "hardest_level_passed": hardest,
              "strong_topics": strong_topics, "weak_topics": weak_topics,
              "questions": questions, "study_plan": plan}
    llm_text = _llm_narrative(report)
    report["summary"] = llm_text or _rule_based_narrative(report)
    report["generated_by"] = "LLM + retrieved context" if llm_text else "offline rules + retrieved context"
    return report


# --------------------------------------------------------------------------------------
# Service layer used by app.py
# --------------------------------------------------------------------------------------
class InterviewFinished(Exception):
    pass


def list_careers():
    out = []
    for career in CAREER_MAP:
        sub = knn_df[knn_df["Career"] == career]
        out.append({"name": career, "questions": int(len(sub)),
                    "difficulty_counts": {lvl: int((sub["Difficulty"] == lvl).sum()) for lvl in DIFFICULTY_LEVELS}})
    return out


def resolve_career(name):
    for career in CAREER_MAP:
        if career.casefold() == str(name).strip().casefold():
            return career
    return None


def _public_question(row, number, total):
    """Question as sent to the app. The expected answer is NOT included."""
    return {"number": number, "total": total, "question": row["Question"],
            "difficulty": row["Difficulty"], "category": row["Category"], "career": row["Career"]}


def _knn_payload(info):
    if not info:
        return None
    return {"based_on": info["topic"], "wanted_difficulty": info["target"], "used_difficulty": info["level"],
            "candidates": [{"question": r["Question"], "difficulty": r["Difficulty"], "category": r["Category"],
                            "similarity": round(float(r["Similarity"]), 3)}
                           for _, r in info["candidates"].iterrows()]}


class LiveInterview:
    """One running interview for the API. Held in memory only."""

    def __init__(self, career, max_questions=10, seed=None):
        self.career = career
        self.max_questions = max_questions
        self.session = InterviewSession(career=career, max_questions=max_questions)
        self.current = self.session.first_question(seed=seed)
        self.created = time.time()
        self._report = None

    @property
    def finished(self):
        return self.current is None or self.session.finished

    def current_question(self):
        if self.current is None:
            return None
        return _public_question(self.current, len(self.session.asked_questions) + 1, self.max_questions)

    def progress(self):
        scores = self.session.scores
        return {"answered": len(scores), "total": self.max_questions,
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
                "current_difficulty": self.session.current_difficulty}

    def answer(self, user_answer):
        if self.finished:
            raise InterviewFinished("The interview is finished. Request /feedback.")
        if not isinstance(user_answer, str):
            raise ValueError("'answer' must be text.")
        if len(user_answer) > MAX_ANSWER_CHARS:
            raise ValueError(f"'answer' is too long (max {MAX_ANSWER_CHARS} characters).")

        row = self.current
        result = score_answer(row["Question"], user_answer)
        self.session.record_answer(row["Question"], result["score"], user_answer=user_answer,
                                   expected_answer=row["Answer"])

        knn = None
        if self.session.can_continue():
            self.current = self.session.next_question(topic=row["Question"])
            knn = _knn_payload(self.session.last_knn) if self.current is not None else None
        else:
            self.current = None
        self._report = None

        return {
            "result": {k: result[k] for k in ("score", "similarity", "keyword_coverage", "matched_keywords",
                                              "missing_keywords", "expected_answer", "feedback")},
            "progress": self.progress(),
            "finished": self.finished,
            "next_question": self.current_question(),
            "knn": knn,
        }

    def end(self):
        self.session.finished = True
        self.current = None
        self._report = None

    def report(self):
        if self._report is None:
            self._report = build_report(self.session.final_summary())
        return self._report
