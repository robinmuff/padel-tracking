"""
Dauer einer Analyse vorab schaetzen.

Grundlage sind die Messwerte frueherer Analysen auf diesem Rechner (result/*/timing.json).
Solange es keine gibt, gelten Richtwerte, gemessen auf einem Apple-Silicon-Mac mit
Neural Engine (Core ML), 720p, 30 fps.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics

import cv2

import padel

# Richtwerte (Sekunden Rechenzeit pro Sekunde Video, bzw. Sekunden)
DEFAULTS = {
    "detect_per_active_s": 1.16,   # Erkennung, Spielererkennung jedes 2. Bild, Core ML
    "detect_per_active_s_torch": 3.9,
    "active_ratio": 0.78,          # Anteil, der bei Pausen-Sprung voll analysiert wird
    "eval_per_s": 0.02,
    "render_per_s": 0.15,          # annotiertes Video + Umwandlung nach H.264
    "load_s": 15.0,
    "convert_s": 60.0,             # einmalige Umwandlung der Modelle fuer die Neural Engine
    "download_bytes_per_s": 12e6,
}
# Kosten pro Frame (ms): TrackNet 35, YOLO 20 pro Lauf -> Skalierung fuer "jedes n-te Bild"
TRACKNET_MS, YOLO_MS = 35.0, 20.0

_info_cache: dict[str, dict] = {}


def _player_factor(player_every):
    k = max(int(player_every or 2), 1)
    return (TRACKNET_MS + YOLO_MS / k) / (TRACKNET_MS + YOLO_MS / 2)


def learned(result_dir, coreml):
    """Mediane der eigenen Messungen (nur Laeufe mit mind. 1 Minute Video)."""
    runs = []
    for path in glob.glob(os.path.join(result_dir, "*", "timing.json")):
        try:
            with open(path, encoding="utf-8") as f:
                t = json.load(f)
        except (OSError, ValueError):
            continue
        if t.get("processed_s", 0) >= 60 and bool(t.get("coreml")) == coreml:
            runs.append(t)
    out, n = {}, len(runs)
    detect = [t["detect_s"] / t["active_s"] / _player_factor(t.get("player_every"))
              for t in runs if t.get("active_s")]
    if detect:
        out["detect_per_active_s"] = statistics.median(detect)
    active = [t["active_s"] / t["processed_s"] for t in runs if t.get("skip_pauses")]
    if active:
        out["active_ratio"] = statistics.median(active)
    evals = [t["eval_s"] / t["range_s"] for t in runs if t.get("eval_s") is not None and t.get("range_s")]
    if evals:
        out["eval_per_s"] = statistics.median(evals)
    renders = [t["render_s"] / t["range_s"] for t in runs if t.get("render_s") and t.get("range_s")]
    if renders:
        out["render_per_s"] = statistics.median(renders)
    loads = [t["load_s"] for t in runs if t.get("load_s") is not None]
    if loads:
        out["load_s"] = statistics.median(loads)
    return out, n


def youtube_id(url):
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([\w-]{11})", url)
    return m[1] if m else None


def video_info(source, root, video_dir, height=720):
    """-> dict(duration, cached, size, path). Lokale Dateien direkt, YouTube ueber den
    Cache oder (einmalig, ein paar Sekunden) ueber die Metadaten von yt-dlp."""
    if not padel.is_url(source):
        path = source if os.path.isabs(source) else os.path.join(root, source)
        return {**_file_duration(path), "cached": True, "size": 0, "path": path}
    vid = youtube_id(source)
    cached = os.path.join(video_dir, f"{vid}_00000-ende_{height}p.mp4") if vid else None
    if cached and os.path.exists(cached):
        return {**_file_duration(cached), "cached": True, "size": 0, "path": cached}
    key = vid or source
    if key not in _info_cache:
        from yt_dlp import YoutubeDL
        with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(source, download=False)
        fmts = [f for f in info.get("formats", []) if (f.get("height") or 0) <= height
                and f.get("vcodec", "none") != "none"]
        fmts.sort(key=lambda f: (str(f.get("vcodec", "")).startswith("avc1"), f.get("height") or 0),
                  reverse=True)
        size = next((f.get("filesize") or f.get("filesize_approx") for f in fmts
                     if f.get("filesize") or f.get("filesize_approx")), 0)
        _info_cache[key] = {"duration": float(info.get("duration") or 0), "fps": 30.0,
                            "cached": False, "size": size or 0, "path": None,
                            "live": info.get("live_status") == "is_live"}
    return _info_cache[key]


def _file_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Video nicht lesbar")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return {"duration": frames / fps, "fps": fps}


def estimate(params, root, result_dir, video_dir, model_dir):
    """-> {"total_s", "parts": [(label, s)], "range_s", "basis"}"""
    source = (params.get("video") or "").strip()
    info = video_info(source, root, video_dir)
    if info.get("live"):
        raise ValueError("Der Stream läuft noch live.")
    start = padel.parse_time(params.get("start") or "") or \
        (padel.youtube_start(source) if padel.is_url(source) else None) or 0.0
    end = padel.parse_time(params.get("end") or "") or info["duration"]
    end = min(end, info["duration"]) if info["duration"] else end
    range_s = max(end - start, 0.0)
    if range_s <= 0:
        raise ValueError("Ende liegt vor dem Start.")

    coreml = padel.coreml_available()
    values = dict(DEFAULTS)
    got, n_runs = learned(result_dir, coreml)
    values.update(got)
    per_active = values["detect_per_active_s"] if coreml else values["detect_per_active_s_torch"]
    skip = params.get("skip_pauses", True)
    ratio = values["active_ratio"] if skip else 1.0

    parts = []
    if not info["cached"]:
        size = info["size"] or info["duration"] * 1.1e5      # ~110 KB/s bei 720p
        parts.append(("Download", size / values["download_bytes_per_s"] + 10))
    if coreml and not (os.path.exists(os.path.join(model_dir, "ball_tracknet.mlpackage"))
                       and os.path.exists(os.path.join(model_dir, "yolov8s.mlpackage"))):
        parts.append(("Modelle umwandeln (einmalig)", values["convert_s"]))
    parts.append(("Start", values["load_s"]))
    parts.append(("Erkennung", range_s * ratio * per_active * _player_factor(params.get("player_every"))))
    parts.append(("Auswertung", range_s * values["eval_per_s"] + 5))
    if params.get("render_video"):
        parts.append(("Video rendern", range_s * values["render_per_s"]))
    total = sum(s for _, s in parts)
    basis = (f"gemessen aus {n_runs} eigenen Analyse{'n' if n_runs != 1 else ''}" if n_runs
             else "Richtwert (wird mit jeder Analyse genauer)")
    return {"total_s": round(total), "parts": [(l, round(s)) for l, s in parts],
            "range_s": round(range_s), "basis": basis}
