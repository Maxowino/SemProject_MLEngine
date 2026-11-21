"""
test_api.py - checks the API end to end.

  python test_api.py                 runs the checks in-process (no server needed)
  python test_api.py --live          runs the same checks against a running server (http://localhost:5000)
  python test_api.py --live http://192.168.1.5:5000
"""
import json
import os
import sys
import urllib.error
import urllib.request

import engine   # only used to look up expected answers so the test can type good answers

LIVE = "--live" in sys.argv
BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:5000")

if LIVE:
    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
else:
    import app as api
    _client = api.app.test_client()

    def call(method, path, body=None):
        r = _client.open(path, method=method, json=body)
        return r.status_code, r.get_json()

HERE = os.path.dirname(os.path.abspath(__file__))
def snapshot():
    return {os.path.join(d, f) for d, dirs, files in os.walk(HERE) if "__pycache__" not in d
            for f in files}

results = []
def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    results.append(bool(ok))

before = snapshot()

# --- basics --------------------------------------------------------------------------
st, body = call("GET", "/health")
check("GET /health", st == 200 and body["status"] == "ok")
st, body = call("GET", "/careers")
check("GET /careers lists 5 careers", st == 200 and len(body["careers"]) == 5)
st, _ = call("GET", "/nope")
check("Unknown route returns JSON 404", st == 404)

# --- /start validation -----------------------------------------------------------------
check("/start rejects an unknown career", call("POST", "/start", {"career": "Cooking"})[0] == 400)
check("/start rejects a missing career", call("POST", "/start", {})[0] == 400)
check("/start rejects bad max_questions", call("POST", "/start", {"career": "Web Development", "max_questions": 99})[0] == 400)

# --- every career can start ---------------------------------------------------------------
for c in [x["name"] for x in call("GET", "/careers")[1]["careers"]]:
    st, body = call("POST", "/start", {"career": c, "max_questions": 3})
    q = body.get("question", {})
    ok = st == 201 and q.get("career") == c and q["question"] in set(engine.knn_df["Question"])
    check(f"/start works for {c}", ok)

# --- full interview ----------------------------------------------------------------------------
st, body = call("POST", "/start", {"career": "Web Development", "max_questions": 10, "seed": 1})
sid, q = body["session_id"], body["question"]
check("/start returns a session and a first question", st == 201 and sid and q["number"] == 1)
check("Expected answer is NOT sent with the question",
      not any(k in q for k in ("answer", "expected_answer", "Answer")))

st, _ = call("GET", f"/feedback/{sid}")
check("/feedback is refused while the interview is running", st == 409)
check("/answer without session_id returns 400", call("POST", "/answer", {"answer": "x"})[0] == 400)
check("/answer with unknown session returns 404", call("POST", "/answer", {"session_id": "zzz", "answer": "x"})[0] == 404)
check("/answer without answer returns 400", call("POST", "/answer", {"session_id": sid})[0] == 400)
check("/answer rejects an over-long answer", call("POST", "/answer", {"session_id": sid, "answer": "a" * 5000})[0] == 400)

asked, levels, last = [], [], None
n = 0
while q is not None:
    n += 1
    asked.append(q["question"]); levels.append(q["difficulty"])
    expected = engine.get_expected_answer(q["question"])
    if n == 1:
        user = ""                                   # blank answer
    elif n % 3 == 0:
        user = "I am not sure about this."          # poor
    elif n % 3 == 1:
        user = expected                             # perfect
    else:
        user = expected[: len(expected) // 2]       # partial
    st, last = call("POST", "/answer", {"session_id": sid, "answer": user})
    if n == 1:
        check("Blank answer scores 0 and still returns the expected answer",
              st == 200 and last["result"]["score"] == 0 and last["result"]["expected_answer"] == expected)
    if st != 200:
        break
    q = last["next_question"]
    if n > 12:
        break

check("Interview finished after exactly 10 questions", n == 10 and last["finished"] is True)
check("All questions unique", len(set(asked)) == 10)
check("All questions come from the dataset", all(x in set(engine.knn_df["Question"]) for x in asked))
check("All questions belong to the chosen career",
      all(engine.knn_df[engine.knn_df["Question"] == x].iloc[0]["Career"] == "Web Development" for x in asked))
check("Difficulty levels are valid", all(l in engine.DIFFICULTY_LEVELS for l in levels))
check("Scores are whole numbers 0-100", last["result"]["score"] == int(last["result"]["score"]) and 0 <= last["result"]["score"] <= 100)
check("Progress is reported", last["progress"]["answered"] == 10)

st, _ = call("POST", "/answer", {"session_id": sid, "answer": "more"})
check("/answer after the end returns 409", st == 409)

st, rep = call("GET", f"/feedback/{sid}")
check("/feedback returns the RAG report", st == 200 and all(k in rep for k in
      ["summary", "study_plan", "questions", "average_score", "weak_topics", "strong_topics"]))
check("Report has one item per answer", len(rep["questions"]) == 10)
check("Report states how it was generated", rep["generated_by"].endswith("retrieved context") or "LLM" in rep["generated_by"])
check("/feedback unknown session returns 404", call("GET", "/feedback/zzz")[0] == 404)

# --- adaptive behaviour: a KNN block appears after the first answer --------------------------------
st, body = call("POST", "/start", {"career": "Systems and DevOps", "max_questions": 5, "seed": 2})
sid2, q2 = body["session_id"], body["question"]
st, r = call("POST", "/answer", {"session_id": sid2, "answer": engine.get_expected_answer(q2["question"])})
knn = r["knn"]
check("KNN details are returned after an answer", st == 200 and knn and len(knn["candidates"]) >= 1)
check("First KNN candidate is the next question", knn["candidates"][0]["question"] == r["next_question"]["question"])

# --- /end -----------------------------------------------------------------------------------------------------
st, r = call("POST", "/end", {"session_id": sid2})
check("/end stops the interview", st == 200 and r["finished"] is True)
st, rep2 = call("GET", f"/feedback/{sid2}")
check("/feedback works after /end", st == 200 and rep2["questions_answered"] == 1)

# --- no files written ------------------------------------------------------------------------------------------
if not LIVE:
    check("The API wrote no files (demo files untouched)", snapshot() == before)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
