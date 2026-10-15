#!/usr/bin/env python3
"""
Padel-Analyse Server: Analysen im Browser starten, verfolgen und ansehen.

Starten:   server/start.sh          (oder: .venv/bin/python server/app.py)
Dann:      http://127.0.0.1:7984

Jede Analyse bekommt einen Ordner result/<id>/ mit job.json (Einstellungen),
run.log (Ausgabe von padel.py) und den Ergebnissen. padel.py laeuft als eigener
Prozess weiter, auch wenn der Server beendet wird; der Status wird aus Prozess
und Log abgeleitet.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime

import cv2
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import padel  # noqa: E402

RESULT_DIR = os.path.join(ROOT, "result")
COURT_DIR = os.path.join(ROOT, "courts")
VIDEO_DIR = os.path.join(ROOT, "videos")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")

app = Flask(__name__, static_folder=None)
download_lock = threading.Lock()


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
def job_dir(job_id):
    if not re.fullmatch(r"[\w.-]+", job_id):
        abort(400)
    path = os.path.join(RESULT_DIR, job_id)
    if not os.path.isdir(path):
        abort(404)
    return path


def read_job(job_id):
    path = os.path.join(RESULT_DIR, job_id, "job.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_job(job):
    path = os.path.join(RESULT_DIR, job["id"], "job.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Zombie (beendet, aber noch nicht eingesammelt) zaehlt nicht
    try:
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def log_tail(job_id, size=64_000):
    path = os.path.join(RESULT_DIR, job_id, "run.log")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - size))
        text = f.read().decode("utf-8", "replace")
    # yt-dlp schreibt Fortschritt mit \r in eine Zeile
    return text.replace("\r", "\n")


PROGRESS = re.compile(r"\[(Erkennung|Video)\] (\S+) im Video, (\d+)%, ([\d.]+) fps(?:, Rest ca\. (\S+))?")
DOWNLOAD = re.compile(r"\[download\]\s+([\d.]+)% of")


def job_status(job):
    """Status und Fortschritt aus Prozess und Log ableiten."""
    tail = log_tail(job["id"]).rsplit("=== Start", 1)[-1]   # nur der letzte Lauf
    lines = [l for l in tail.splitlines() if l.strip()]
    running = alive(job.get("pid"))
    status = {"state": "neu", "phase": "", "percent": 0, "eta": "", "fps": None, "message": ""}

    done = any(l.startswith("Gespeichert in") for l in lines)
    error = next((l for l in reversed(lines) if re.match(r"\w*(Error|Exception)\b|Traceback", l)
                  or l.startswith("Video nicht lesbar") or l.startswith("Download fehlgeschlagen")), None)
    if running:
        status["state"] = "läuft"
    elif done:
        status["state"] = "fertig"
    elif job.get("stopped"):
        status["state"] = "gestoppt"
    elif error:
        status["state"] = "Fehler"
        status["message"] = error[:300]
    elif job.get("pid"):
        status["state"] = "abgebrochen"
    elif os.path.exists(os.path.join(RESULT_DIR, job["id"], "dashboard.html")):
        done = True                       # vor dem Server angelegt (ohne Log)
        status["state"] = "fertig"

    for l in reversed(lines):
        m = PROGRESS.search(l)
        if m:
            status.update(phase=m[1], percent=int(m[3]), fps=float(m[4]), eta=m[5] or "",
                          position=m[2])
            break
        m = DOWNLOAD.search(l)
        if m:
            status.update(phase="Download", percent=int(float(m[1])))
            break
        if l.startswith("Ball in "):
            status.update(phase="Auswertung", percent=100, message=l)
            break
        if l.startswith("Lade ") or l.startswith("Video aus Cache"):
            status.update(phase="Vorbereitung")
            break
    if done:
        status.update(phase="Fertig", percent=100, eta="")
        summary = next((l for l in reversed(lines) if l.endswith("Ballwechsel analysiert.")), "")
        status["message"] = summary
    return status


def build_command(job):
    p = job["params"]
    out_dir = os.path.relpath(os.path.join(RESULT_DIR, job["id"]), ROOT)
    args = [PYTHON, "-u", "padel.py", "--video", p["video"], "--court", p["court"],
            "--out-dir", out_dir, "--player-every", str(int(p.get("player_every") or 2))]
    for key in ("start", "end", "revers", "lefties"):
        if p.get(key):
            args += [f"--{key}", str(p[key])]
    if not p.get("render_video", True):
        args.append("--no-video")
    if not p.get("skip_pauses", True):
        args.append("--no-skip")
    return args


def launch(job):
    """padel.py als eigene Prozessgruppe starten; caffeinate haelt den Mac wach."""
    d = os.path.join(RESULT_DIR, job["id"])
    cmd = " ".join(shlex.quote(a) for a in build_command(job))
    log = open(os.path.join(d, "run.log"), "a", encoding="utf-8")
    log.write(f"\n=== Start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n{cmd}\n")
    log.flush()
    wrapper = ["caffeinate", "-i", "/bin/sh", "-c", cmd] if os.path.exists("/usr/bin/caffeinate") \
        else ["/bin/sh", "-c", cmd]
    proc = subprocess.Popen(wrapper, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    job.update(pid=proc.pid, stopped=False, started=datetime.now().isoformat(timespec="seconds"))
    write_job(job)


def running_job():
    for job in all_jobs():
        if alive(job.get("pid")):
            return job
    return None


def all_jobs():
    jobs = []
    if os.path.isdir(RESULT_DIR):
        for name in os.listdir(RESULT_DIR):
            job = read_job(name)
            if job:
                jobs.append(job)
    return sorted(jobs, key=lambda j: j.get("created", ""), reverse=True)


def slug(text):
    text = re.sub(r"[^\w-]+", "-", text.strip().lower()).strip("-")
    return text[:40] or "analyse"


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/jobs")
def api_jobs():
    out = []
    for job in all_jobs():
        d = os.path.join(RESULT_DIR, job["id"])
        out.append({**job, "status": job_status(job),
                    "has_dashboard": os.path.exists(os.path.join(d, "dashboard.html")),
                    "has_video": os.path.exists(os.path.join(d, "analyse.mp4"))})
    return jsonify(out)


@app.post("/api/jobs")
def api_create_job():
    p = request.get_json(force=True)
    if not p.get("video"):
        return jsonify(error="Bitte einen YouTube-Link oder eine Videodatei angeben."), 400
    if not p.get("court") or not os.path.exists(os.path.join(ROOT, p["court"])):
        return jsonify(error="Bitte eine Platz-Kalibrierung wählen (oder neu kalibrieren)."), 400
    for key in ("start", "end"):
        if p.get(key):
            try:
                padel.parse_time(p[key])
            except ValueError:
                return jsonify(error=f"Ungültige Zeit bei {key}: {p[key]}"), 400
    if running_job():
        return jsonify(error="Es läuft bereits eine Analyse. Bitte warten oder stoppen."), 409
    name = p.get("name") or f"Analyse {p.get('start') or ''}".strip()
    job_id = f"{datetime.now():%Y-%m-%d_%H%M}_{slug(name)}"
    os.makedirs(os.path.join(RESULT_DIR, job_id), exist_ok=False)
    params = {k: p.get(k) for k in ("video", "start", "end", "court", "render_video",
                                    "skip_pauses", "player_every", "revers", "lefties")}
    params["render_video"] = bool(p.get("render_video"))
    params["skip_pauses"] = bool(p.get("skip_pauses", True))
    job = {"id": job_id, "name": name, "params": params,
           "created": datetime.now().isoformat(timespec="seconds")}
    write_job(job)
    launch(job)
    return jsonify(job), 201


@app.post("/api/jobs/<job_id>/stop")
def api_stop(job_id):
    job_dir(job_id)
    job = read_job(job_id)
    if alive(job.get("pid")):
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    job["stopped"] = True
    write_job(job)
    return jsonify(ok=True)


@app.post("/api/jobs/<job_id>/resume")
def api_resume(job_id):
    """Gleiche Einstellungen erneut starten; padel.py macht beim Zwischenstand weiter."""
    job_dir(job_id)
    if running_job():
        return jsonify(error="Es läuft bereits eine Analyse."), 409
    job = read_job(job_id)
    p = request.get_json(silent=True) or {}
    # Auswertungs-Einstellungen duerfen sich aendern, ohne die Erkennung zu wiederholen
    for key in ("render_video", "revers", "lefties"):
        if key in p:
            job["params"][key] = p[key]
    launch(job)
    return jsonify(ok=True)


@app.delete("/api/jobs/<job_id>")
def api_delete(job_id):
    import shutil
    d = job_dir(job_id)
    job = read_job(job_id)
    if job and alive(job.get("pid")):
        return jsonify(error="Analyse läuft noch. Bitte zuerst stoppen."), 409
    shutil.rmtree(d)
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>/log")
def api_log(job_id):
    job_dir(job_id)
    return log_tail(job_id, 20_000), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/api/courts")
def api_courts():
    files = sorted(f for f in os.listdir(COURT_DIR) if f.endswith(".json")) if os.path.isdir(COURT_DIR) else []
    return jsonify([{"name": f[:-5], "path": f"courts/{f}"} for f in files])


@app.post("/api/courts")
def api_save_court():
    p = request.get_json(force=True)
    corners = p.get("corners")
    if not corners or len(corners) != 4:
        return jsonify(error="Es braucht genau 4 Ecken."), 400
    name = slug(p.get("name") or "platz")
    os.makedirs(COURT_DIR, exist_ok=True)
    padel.Court(corners).save(os.path.join(COURT_DIR, f"{name}.json"))
    return jsonify(path=f"courts/{name}.json")


@app.get("/api/videos")
def api_videos():
    files = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")) if os.path.isdir(VIDEO_DIR) else []
    return jsonify([f"videos/{f}" for f in files])


def resolve_video(source):
    """YouTube-Link -> (gecachte) lokale Datei; lokale Pfade relativ zum Projekt."""
    if padel.is_url(source):
        with download_lock:
            path, _ = padel.download_video(source, cache_dir=VIDEO_DIR)
        return path
    path = source if os.path.isabs(source) else os.path.join(ROOT, source)
    if not os.path.exists(path):
        abort(404, "Video nicht gefunden")
    return path


def grab_frame(source, t):
    cap = cv2.VideoCapture(resolve_video(source))
    if t is None and padel.is_url(source):
        t = padel.youtube_start(source)
    cap.set(cv2.CAP_PROP_POS_MSEC, (t or 0) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        abort(400, "Kein Bild an dieser Stelle")
    return frame


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), 200, {"Content-Type": "image/jpeg", "Cache-Control": "no-store"}


@app.get("/api/frame")
def api_frame():
    t = request.args.get("t")
    return jpeg(grab_frame(request.args["source"], padel.parse_time(t) if t else None))


@app.post("/api/court-preview")
def api_court_preview():
    """Bild mit den aus den 4 Ecken berechneten Platzlinien, zur Kontrolle."""
    p = request.get_json(force=True)
    frame = grab_frame(p["source"], padel.parse_time(p["t"]) if p.get("t") else None)
    court = padel.Court(p["corners"])
    lines = [((0, 0), (10, 0)), ((10, 0), (10, 20)), ((10, 20), (0, 20)), ((0, 20), (0, 0)),
             ((0, 10), (10, 10)), ((0, 3.05), (10, 3.05)), ((0, 16.95), (10, 16.95)),
             ((5, 3.05), (5, 16.95))]
    for a, b in lines:
        cv2.line(frame, court.to_image(a), court.to_image(b), (0, 0, 255), 2)
    return jpeg(frame)


# --------------------------------------------------------------------------
# Seiten und Dateien (send_file beherrscht Range-Requests -> Video-Spulen geht)
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/calibrate")
def calibrate():
    return send_from_directory(STATIC_DIR, "calibrate.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC_DIR, name)


@app.get("/result/<path:name>")
def result_files(name):
    return send_from_directory(RESULT_DIR, name, conditional=True)


@app.get("/videos/<path:name>")
def video_files(name):
    return send_from_directory(VIDEO_DIR, name, conditional=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Padel-Analyse Server")
    ap.add_argument("--port", type=int, default=7984)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Padel-Analyse laeuft auf {url}  (Beenden: Ctrl+C)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True)
