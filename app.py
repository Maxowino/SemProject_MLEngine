"""
app.py - Flask API for the IT Interview Assistant.

Endpoints
  GET  /health
  GET  /careers
  POST /start      {"career": "...", "max_questions": 10}      -> session_id + first question
  POST /answer     {"session_id": "...", "answer": "..."}      -> score, feedback, next question
  POST /end        {"session_id": "..."}                       -> stop early
  GET  /feedback/<session_id>                                  -> RAG feedback report (after the interview ends)

Run:  python app.py        (then open http://localhost:5000/health)
Safe for the VS Code demo: sessions live in memory and no files are written.
"""
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from werkzeug.exceptions import HTTPException

import engine

MAX_SESSIONS = 500
SESSION_TTL_SECONDS = 2 * 60 * 60


class _JSONProvider(DefaultJSONProvider):
    @staticmethod
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


app = Flask(__name__)
app.json = _JSONProvider(app)

SESSIONS = {}
_LOCK = threading.Lock()


def _error(message, status):
    return jsonify({"error": message}), status


def _cleanup():
    now = time.time()
    for sid in [s for s, live in SESSIONS.items() if now - live.created > SESSION_TTL_SECONDS]:
        del SESSIONS[sid]
    while len(SESSIONS) >= MAX_SESSIONS:
        del SESSIONS[min(SESSIONS, key=lambda s: SESSIONS[s].created)]


def _get_session(data):
    sid = (data or {}).get("session_id")
    if not sid:
        return None, _error("'session_id' is required.", 400)
    live = SESSIONS.get(sid)
    if live is None:
        return None, _error("Unknown or expired session_id. Call /start again.", 404)
    return live, None


@app.get("/")
@app.get("/health")
def health():
    return jsonify({"status": "ok", "questions_in_dataset": int(len(engine.knn_df)),
                    "active_sessions": len(SESSIONS)})


@app.get("/careers")
def careers():
    return jsonify({"careers": engine.list_careers()})


@app.post("/start")
def start():
    data = request.get_json(silent=True) or {}
    career = engine.resolve_career(data.get("career", ""))
    if career is None:
        return _error("Unknown career. Choose one of: " + ", ".join(engine.CAREER_MAP), 400)

    max_q = data.get("max_questions", 10)
    if not isinstance(max_q, int) or isinstance(max_q, bool) or not 1 <= max_q <= 20:
        return _error("'max_questions' must be a whole number from 1 to 20.", 400)

    seed = data.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        return _error("'seed' must be a whole number.", 400)

    live = engine.LiveInterview(career, max_questions=max_q, seed=seed)
    if live.current is None:
        return _error("No questions available for this career.", 404)

    with _LOCK:
        _cleanup()
        sid = uuid.uuid4().hex
        SESSIONS[sid] = live

    return jsonify({"session_id": sid, "career": career, "max_questions": max_q,
                    "note": "Cold start: a random dataset question; KNN takes over after the first answer.",
                    "question": live.current_question()}), 201


@app.post("/answer")
def answer():
    data = request.get_json(silent=True) or {}
    live, err = _get_session(data)
    if err:
        return err
    if "answer" not in data:
        return _error("'answer' is required.", 400)
    try:
        with _LOCK:
            return jsonify(live.answer(data["answer"]))
    except engine.InterviewFinished as e:
        return _error(str(e), 409)
    except ValueError as e:
        return _error(str(e), 400)


@app.post("/end")
def end():
    live, err = _get_session(request.get_json(silent=True) or {})
    if err:
        return err
    with _LOCK:
        live.end()
    return jsonify({"finished": True, "progress": live.progress()})


@app.get("/feedback/<session_id>")
def feedback(session_id):
    live = SESSIONS.get(session_id)
    if live is None:
        return _error("Unknown or expired session_id.", 404)
    if not live.finished:
        return _error("The interview is still running. Finish it or call /end first.", 409)
    return jsonify(live.report())


@app.errorhandler(404)
def _not_found(_):
    return _error("Not found.", 404)


@app.errorhandler(405)
def _bad_method(_):
    return _error("Method not allowed.", 405)


@app.errorhandler(Exception)
def _unexpected(e):
    if isinstance(e, HTTPException):
        return _error(e.description, e.code)
    app.logger.exception(e)
    return _error("Server error. Check the terminal running app.py.", 500)


if __name__ == "__main__":
    # 0.0.0.0 lets the Android emulator (10.0.2.2) and a phone on the same Wi-Fi reach the server
    app.run(host="0.0.0.0", port=5000, debug=False)
