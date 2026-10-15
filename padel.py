#!/usr/bin/env python3
"""
padel.py - Ballwechsel verfolgen, Schlaege und Fehler erkennen.

Gedacht fuer Aufnahmen aus erhoehter Position hinter einer Grundlinie
(typische Club- oder TV-Hauptkamera). Wiederholungen, Zooms und
Kamerawechsel muessen vorher herausgeschnitten werden.

Installation:
    pip install ultralytics opencv-python numpy yt-dlp lap ocrmac
    brew install ffmpeg        # nur fuer --start/--end bei YouTube-Links

Verwendung:
    # 1. Erster Lauf: Platzecken anklicken (wird in court.json gespeichert)
    python padel.py --video match.mp4 --calibrate

    # 2. Weitere Laeufe mit gespeicherter Kalibrierung
    python padel.py --video match.mp4 --court courts/milano-court2.json --show

    # YouTube-Link: nur den Ausschnitt laden (Cache in videos/)
    python padel.py --video "https://www.youtube.com/live/..." --start 1:05:00 --end 1:20:00 --calibrate

    # Lokale Datei: Ausschnitt analysieren
    python padel.py --video match.mp4 --start 30:00 --end 45:00 --court courts/milano-court2.json

Ablauf (offline, drei Durchgaenge):
    1. Pro Frame: Spieler (YOLO + ByteTrack) und Ball-Wahrscheinlichkeit (TrackNet)
    2. Ballbahn aus den Kandidaten verketten, Schlaege = Richtungswechsel am Spieler
    3. Annotiertes Video schreiben

Ausgabe (Ordner result/, aenderbar mit --out-dir):
    analyse.mp4     annotiertes Video
    events.json     alle Ballwechsel mit Schlaegen und Ergebnis
    stats.csv       Statistik pro Spieler
    dashboard.html  Video mit klickbarer Schlagliste (im Browser oeffnen)

Koordinaten: x = 0..10 m (Breite), y = 0..20 m (Laenge).
y = 0 ist die ferne Grundlinie, y = 20 die nahe, Netz bei y = 10.
Team A = nahe Seite (unten im Bild), Team B = ferne Seite.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

COURT_W, COURT_L, NET_Y = 10.0, 20.0, 10.0
NET_ZONE = 4.0  # Meter: naeher am Netz gilt als "Netzposition"

# TrackNet-Gewichte (Tennis/Padel-Ball, 640x360, 3 Frames) aus dem Padex-Release
TRACKNET_URL = ("https://github.com/rondo-labs/Padex/releases/download/v0.1.0/"
                "ball_detection_TrackNet.pt")
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


def download_once(url, path):
    """Gewichte beim ersten Lauf nach models/ laden."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Lade {url} -> {path}")
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)


# --------------------------------------------------------------------------
# Platz-Kalibrierung
# --------------------------------------------------------------------------
class Court:
    """Homographie Bildpixel <-> Platzkoordinaten in Metern."""

    def __init__(self, image_pts):
        self.image_pts = np.array(image_pts, dtype=np.float32)
        court_pts = np.array(
            [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32
        )
        self.H = cv2.getPerspectiveTransform(self.image_pts, court_pts)
        self.H_inv = np.linalg.inv(self.H)

    def to_court(self, pt):
        p = np.array([[pt]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(p, self.H)[0, 0]
        return float(x), float(y)

    def to_image(self, pt):
        p = np.array([[pt]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(p, self.H_inv)[0, 0]
        return int(round(x)), int(round(y))

    @staticmethod
    def inside(pt, margin=0.0):
        x, y = pt
        return -margin <= x <= COURT_W + margin and -margin <= y <= COURT_L + margin

    @staticmethod
    def team_of(pt):
        return "A" if pt[1] > NET_Y else "B"

    def draw(self, img):
        cv2.polylines(img, [self.image_pts.astype(int)], True, (255, 255, 255), 1)
        cv2.line(img, self.to_image((0, NET_Y)), self.to_image((COURT_W, NET_Y)),
                 (200, 200, 200), 2)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"corners": self.image_pts.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["corners"])


def calibrate_interactive(frame) -> Court:
    labels = ["fern links", "fern rechts", "nah rechts", "nah links"]
    pts = []
    win = "Ecken klicken: fern links, fern rechts, nah rechts, nah links (ESC = Abbruch)"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while len(pts) < 4:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, labels[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, disp)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow(win)
    if len(pts) < 4:
        raise SystemExit("Kalibrierung abgebrochen.")
    return Court(pts)


# --------------------------------------------------------------------------
# Spieler
# --------------------------------------------------------------------------
@dataclass
class Player:
    track_id: int
    box: tuple          # x1, y1, x2, y2 in Pixeln
    court_pos: tuple    # Fussposition in Metern
    name: str = ""

    @property
    def team(self):
        return Court.team_of(self.court_pos)


class PlayerTracker:
    """YOLO-Personenerkennung + ByteTrack, gefiltert auf die 4 Spieler im Feld.
    imgsz=1280, damit die kleinen Spieler hinter dem fernen Gitter erkannt werden."""

    def __init__(self, court: Court, model_path="models/yolov8s.pt", conf=0.25, imgsz=1280,
                 coreml=True):
        from ultralytics import YOLO
        download_once(YOLO_URL + os.path.basename(model_path), model_path)
        self.imgsz = imgsz
        if coreml and coreml_available():
            # Apple Neural Engine: ~3.5x schneller als PyTorch auf der GPU
            ml_path = os.path.splitext(model_path)[0] + ".mlpackage"
            self.imgsz = (round(imgsz * 9 / 16 / 32) * 32, imgsz)      # 16:9, durch 32 teilbar
            if not os.path.exists(ml_path):
                print("Wandle YOLO fuer die Neural Engine um (einmalig) ...")
                YOLO(model_path).export(format="coreml", imgsz=self.imgsz, half=True, nms=False)
            model_path = ml_path
        self.model = YOLO(model_path, task="detect")
        self.court = court
        self.conf = conf
        self.names: dict[int, str] = {}

    def update(self, frame) -> list[Player]:
        r = self.model.track(frame, persist=True, classes=[0], conf=self.conf,
                             imgsz=self.imgsz, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        players = []
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            x1, y1, x2, y2 = map(float, box)
            pos = self.court.to_court(((x1 + x2) / 2, y2))
            if Court.inside(pos, margin=1.5):
                players.append(Player(tid, (x1, y1, x2, y2), pos))

        # Pro Team max. 2 Personen (die groessten) -> Schiri, Ballkinder fliegen raus
        result = []
        for team in "AB":
            tp = [p for p in players if p.team == team]
            tp.sort(key=lambda p: -(p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
            result += tp[:2]

        # Namen pro Track-ID vergeben; die beiden Spieler eines Teams heissen immer
        # verschieden (L/R relativ zueinander, sonst nach Platzhaelfte)
        for team in "AB":
            tp = sorted((p for p in result if p.team == team), key=lambda p: p.court_pos[0])
            known = {self.names[p.track_id] for p in tp if p.track_id in self.names}
            for i, p in enumerate(tp):
                if p.track_id in self.names:
                    continue
                if len(tp) == 2 and not known:
                    side = "LR"[i]
                elif known:
                    side = "R" if f"{team}-L" in known else "L"
                else:
                    side = "L" if p.court_pos[0] < COURT_W / 2 else "R"
                self.names[p.track_id] = f"{team}-{side}"
                known.add(self.names[p.track_id])
            for p in tp:
                p.name = self.names[p.track_id]
        return result


# --------------------------------------------------------------------------
# Ball: TrackNet liefert pro Frame Kandidaten mit Wahrscheinlichkeit
# --------------------------------------------------------------------------
def _build_tracknet():
    """VGG-artiges Encoder/Decoder-Netz (TrackNet). Eingabe: 3 Frames RGB gestapelt,
    640x360. Ausgabe: 256 Klassen pro Pixel; Klasse 0 = kein Ball."""
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                       nn.ReLU(inplace=True), nn.BatchNorm2d(cout))

        def forward(self, x):
            return self.block(x)

    channels = [(9, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256),
                (256, 256), (256, 512), (512, 512), (512, 512), (512, 256), (256, 256),
                (256, 256), (256, 128), (128, 128), (128, 64), (64, 64), (64, 256)]

    class TrackNet(nn.Module):
        def __init__(self):
            super().__init__()
            for i, (cin, cout) in enumerate(channels, 1):
                setattr(self, f"conv{i}", Block(cin, cout))

        def forward(self, x):
            c = lambda i, t: getattr(self, f"conv{i}")(t)
            up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=True)
            x = F.max_pool2d(c(2, c(1, x)), 2)
            x = F.max_pool2d(c(4, c(3, x)), 2)
            x = F.max_pool2d(c(7, c(6, c(5, x))), 2)
            x = c(10, c(9, c(8, x)))
            x = c(13, c(12, c(11, up(x))))
            x = c(15, c(14, up(x)))
            return c(18, c(17, c(16, up(x))))

    return TrackNet()


def coreml_available():
    if sys.platform != "darwin":
        return False
    try:
        import coremltools  # noqa: F401
        return True
    except ImportError:
        return False


class _BallProb:
    """TrackNet mit P(Ball) pro Pixel als Ausgabe, halbe Aufloesung (180x320).
    Die 256-Klassen-Ausgabe wird im Netz zusammengefasst -> viel weniger Daten."""

    def __new__(cls, net):
        import torch

        class BallProb(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                p = 1 - torch.softmax(self.net(x), dim=1)[:, 0]
                return p[:, ::2, ::2]

        return BallProb()


class BallDetector:
    W, H = 640, 360

    def __init__(self, model_path="models/ball_tracknet.pt", min_prob=0.08, max_peaks=3,
                 coreml=True):
        import torch
        download_once(TRACKNET_URL, model_path)
        self.torch = torch
        net = _build_tracknet()
        net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model = _BallProb(net).eval()
        self.coreml = None
        if coreml and coreml_available():
            import coremltools as ct
            ml_path = os.path.splitext(model_path)[0] + ".mlpackage"
            if not os.path.exists(ml_path):
                print("Wandle TrackNet fuer die Neural Engine um (einmalig) ...")
                example = torch.rand(1, 9, self.H, self.W)
                traced = torch.jit.trace(self.model, example)
                ct.convert(traced, inputs=[ct.TensorType(name="frames", shape=example.shape)],
                           outputs=[ct.TensorType(name="prob")],
                           minimum_deployment_target=ct.target.macOS14,
                           compute_precision=ct.precision.FLOAT16).save(ml_path)
            self.coreml = ct.models.MLModel(ml_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        else:
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
            self.model.to(self.device)
        self.min_prob, self.max_peaks = min_prob, max_peaks
        self.buf = []

    def reset(self):
        """Nach einem Sprung im Video: alte Frames gehoeren nicht mehr dazu."""
        self.buf = []

    def peaks(self, frame):
        """Bis zu max_peaks Ballkandidaten (prob, x, y) in Originalpixeln."""
        self.buf = (self.buf + [cv2.resize(frame, (self.W, self.H))])[-3:]
        if len(self.buf) < 3:
            return []
        x = np.concatenate(self.buf[::-1], axis=2).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        if self.coreml is not None:
            prob = self.coreml.predict({"frames": np.ascontiguousarray(x)})["prob"][0]
        else:
            with self.torch.no_grad():
                prob = self.model(self.torch.from_numpy(x).to(self.device))[0].cpu().numpy()
        prob = prob.astype(np.float32)                 # 320x180 reicht fuer Peaks
        local_max = prob == cv2.dilate(prob, np.ones((7, 7), np.uint8))
        ys, xs = np.where(local_max & (prob > self.min_prob))
        sx, sy = frame.shape[1] / prob.shape[1], frame.shape[0] / prob.shape[0]
        found = sorted(((float(prob[y, x]), (x + 0.5) * sx, (y + 0.5) * sy)
                        for y, x in zip(ys, xs)), reverse=True)
        return found[:self.max_peaks]


def build_ball_path(peaks, gate=45.0, max_gap=6, min_len=4, min_score=1.2, bridge=10):
    """Verkettet Kandidaten zu Teilbahnen (Tracklets) und waehlt die plausibelsten,
    zeitlich nicht ueberlappenden aus. Kurze Luecken werden linear gefuellt.
    peaks: Liste (pro Frame) von [(prob, x, y), ...] -> dict frame -> (x, y)"""
    tracks, active = [], []
    for f, cands in enumerate(peaks):
        used = set()
        for t in sorted(active, key=lambda t: -t["score"]):
            (lf, lx, ly), vx, vy = t["pts"][-1], t["vx"], t["vy"]
            gap = f - lf
            px, py = lx + vx * gap, ly + vy * gap
            best, best_d = None, gate * (1 + 0.5 * (gap - 1))
            for i, (p, x, y) in enumerate(cands):
                d = math.hypot(x - px, y - py)
                if i not in used and d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                p, x, y = cands[best]
                t["vx"], t["vy"] = (x - lx) / gap, (y - ly) / gap
                t["pts"].append((f, x, y))
                t["score"] += p
        for i, (p, x, y) in enumerate(cands):
            if i not in used:
                t = {"pts": [(f, x, y)], "vx": 0.0, "vy": 0.0, "score": p}
                active.append(t)
                tracks.append(t)
        active = [t for t in active if f - t["pts"][-1][0] < max_gap]

    good = [t for t in tracks if len(t["pts"]) >= min_len and t["score"] >= min_score]
    good.sort(key=lambda t: -t["score"])
    taken, path = [], {}
    for t in good:
        a, b = t["pts"][0][0], t["pts"][-1][0]
        if any(a <= tb and ta <= b for ta, tb in taken):
            continue
        taken.append((a, b))
        for f, x, y in t["pts"]:
            path[f] = (x, y)

    frames = sorted(path)
    for f0, f1 in zip(frames, frames[1:]):
        if 1 < f1 - f0 <= bridge:
            (x0, y0), (x1, y1) = path[f0], path[f1]
            for f in range(f0 + 1, f1):
                t = (f - f0) / (f1 - f0)
                path[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return path


# --------------------------------------------------------------------------
# Schlaege: Richtungswechsel der Ballbahn in Spielernaehe
# --------------------------------------------------------------------------
def player_near(pt, players, reach=0.6):
    """Spieler, dessen (um Schlaegerreichweite erweiterte) Box den Ball enthaelt."""
    best, best_d = None, 1e9
    for p in players:
        x1, y1, x2, y2 = p.box
        m = reach * (y2 - y1)
        if x1 - m <= pt[0] <= x2 + m and y1 - m <= pt[1] <= y2 + 0.2 * m:
            d = math.dist(pt, ((x1 + x2) / 2, (y1 + y2) / 2))
            if d < best_d:
                best, best_d = p, d
    return best


def _velocity(path, f, direction, k=7):
    """Geschwindigkeit (px/Frame) als Geradenfit ueber k Frames vor bzw. nach f.
    Der Fit glaettet das Ruckeln durch doppelte Frames (25->30 fps-Streams)."""
    pts = [(g, *path[g]) for g in range(f, f + direction * (k + 1), direction) if g in path]
    if len(pts) < 3:
        return None
    a = np.array(pts, dtype=float)
    return np.polyfit(a[:, 0], a[:, 1], 1)[0], np.polyfit(a[:, 0], a[:, 2], 1)[0]


@dataclass
class Hit:
    frame: int
    team: str
    player: Optional[Player] = None   # None = Schlag nicht sichtbar, nur erschlossen


def away_from_hitter(p, v_out, min_speed=0.5):
    """Ein Schlag der nahen Seite schickt den Ball Richtung Gegner, also im Bild nach
    oben (weg von der Kamera, und meist auch steigend). Ein Ball, der an einer nahen
    Spielerin vorbei nach unten weiterfliegt, wurde nicht geschlagen. Fuer die ferne
    Seite gibt es keine solche Regel: ein Lob steigt im Bild zuerst ebenfalls."""
    return p.team != "A" or v_out[1] < -min_speed


def detect_hits(path, players_per_frame, fps, min_kink=0.9, nms=8, new_track_gap=8,
                double_hit_s=1.5):
    """Ein Schlag aendert die Ballgeschwindigkeit sprunghaft, und zwar dort, wo ein
    Spieler steht. Zusaetzlich zaehlt der Beginn einer Ballspur am Spieler
    (Aufschlag, oder der Ball war beim Treffen vom Koerper verdeckt).
    Die Knickstaerke wird auf die Spielergroesse bezogen (Perspektive: ferne
    Spieler und Ballwege sind im Bild nur halb so gross)."""
    cands = []
    for f in sorted(path):
        v_in, v_out = _velocity(path, f, -1), _velocity(path, f, +1)
        if v_out is None:
            continue
        if v_in is None:
            # Spur beginnt: Ball wird erst ein paar Frames nach dem Treffen sichtbar
            # -> Flugbahn zurueckrechnen und den Spieler am Treffpunkt suchen
            if any(g in path for g in range(f - new_track_gap, f)):
                continue
            for k in range(new_track_gap):
                back = (path[f][0] - k * v_out[0], path[f][1] - k * v_out[1])
                p = player_near(back, players_per_frame[max(f - k, 0)])
                if p is not None and away_from_hitter(p, v_out):
                    cands.append((10.0, f, p))
                    break
            continue
        p = player_near(path[f], players_per_frame[f])
        if p is None or not away_from_hitter(p, v_out):
            continue
        # v_in wurde rueckwaerts gefittet, die Steigung ist trotzdem die Vorwaertsgeschw.
        kink = 10 * math.hypot(v_out[0] - v_in[0], v_out[1] - v_in[1]) / (p.box[3] - p.box[1])
        if kink >= min_kink:
            cands.append((kink, f, p))

    # Non-Maximum-Suppression: pro Zeitfenster nur der staerkste Knick
    hits = []
    for kink, f, p in sorted(cands, key=lambda c: -c[0]):
        if all(abs(f - g) > nms for _, g, _ in hits):
            hits.append((kink, f, p))
    hits.sort(key=lambda h: h[1])

    def at_feet(f, p):
        y1, y2 = p.box[1], p.box[3]
        return path[f][1] > y2 - 0.25 * (y2 - y1)

    # Gleiches Team zweimal hintereinander:
    # - kurz nacheinander = Aufsprung + Schlag oder Doppelerkennung: ein Kontakt auf
    #   Fusshoehe ist der Aufsprung, sonst zaehlt der staerkere Knick
    # - mit Abstand = der Gegner hat dazwischen geschlagen, war aber nicht sichtbar
    merged: list[tuple] = []
    for kink, f, p in hits:
        prev = merged[-1] if merged else None
        if prev and prev[2].team == p.team:
            if f - prev[1] <= double_hit_s * fps:
                if at_feet(prev[1], prev[2]) or (kink > prev[0] and not at_feet(f, p)):
                    merged[-1] = (kink, f, p)
                continue
            gaps = [g for g in range(prev[1] + 1, f) if g not in path]
            merged.append((0.0, gaps[len(gaps) // 2] if gaps else (prev[1] + f) // 2, None))
        merged.append((kink, f, p))
    result, last_team = [], None
    for _, f, p in merged:
        team = p.team if p else other(last_team)
        result.append(Hit(f, team, p))
        last_team = team
    return result


# --------------------------------------------------------------------------
# Ballwechsel-Logik
# --------------------------------------------------------------------------
@dataclass
class Shot:
    frame: int                # Frame ab Beginn des Ausschnitts (= Zeit im analyse.mp4)
    time: str                 # Zeit im Originalvideo
    player: str               # Platzseite, z.B. A-L (A = nah, B = fern)
    team: str
    shot_type: str
    hand: str
    court_x: Optional[float] = None   # Position des Schlaegers in Metern
    court_y: Optional[float] = None
    team_name: str = ""       # vom Scoreboard, z.B. "Borrero / Navarro"
    role: str = ""            # Revés (linke Seite aus eigener Sicht) oder Drive
    player_name: str = ""     # Spielerin, z.B. "Navarro" oder "Borrero / Navarro (Revés)"


@dataclass
class Rally:
    id: int
    start_frame: int
    end_frame: int = -1
    start_time: str = ""
    shots: list = field(default_factory=list)
    result: str = ""
    winner_team: str = ""
    decided_by: str = ""      # Spieler, der den Punkt gewonnen oder den Fehler gemacht hat
    team_names: dict = field(default_factory=dict)   # {"A": "...", "B": "..."}
    score_before: str = ""
    score_after: str = ""
    winner_name: str = ""


def classify_shot(player: Player, ball_pt, is_serve):
    x1, y1, x2, y2 = player.box
    rel_h = (y2 - ball_pt[1]) / max(y2 - y1, 1)   # 0 = Fuesse, 1 = Kopf
    dist_net = abs(player.court_pos[1] - NET_Y)

    # Vorhand/Rueckhand: Team A steht mit dem Ruecken zur Kamera, Team B gespiegelt
    ball_right = ball_pt[0] > (x1 + x2) / 2
    if player.team == "B":
        ball_right = not ball_right   # Linkshaenderinnen: siehe apply_scoreboard
    hand = "Vorhand" if ball_right else "Rueckhand"

    if is_serve:
        return "Aufschlag", hand
    # Bildhoehe statt echter Hoehe: ein anfliegender Ball wirkt hoeher als er ist,
    # deshalb erst deutlich ueber dem Kopf als Ueberkopf werten
    if rel_h > 1.3:
        return ("Smash" if dist_net <= NET_ZONE else "Bandeja/Vibora"), "Ueberkopf"
    if dist_net <= NET_ZONE:
        return "Volley", hand
    return "Grundschlag", hand


def other(team):
    return "B" if team == "A" else "A"


def role_of(label):
    """Seite im Bild -> Position aus Sicht der Spielerin. Team A steht mit dem Ruecken
    zur Kamera (Bild links = ihre linke Seite), Team B schaut zur Kamera (gespiegelt).
    Profis bleiben das ganze Match auf ihrer Seite: links = Revés, rechts = Drive."""
    team, side = label[0], label[-1]
    if side not in "LR":
        return ""
    return "Revés" if (side == "L") == (team == "A") else "Drive"


def build_rallies(hits, path, court, fps, offset_s, max_pause_s=4.0):
    """Schlaege zu Ballwechseln gruppieren. Ein Ballwechsel endet, wenn laenger als
    max_pause_s kein Schlag folgt. Ergebnis aus der letzten Ballposition:
    Ball endet beim Gegner -> Punkt fuer den letzten Schlaeger (Winner),
    Ball endet auf eigener Seite -> Fehler (Netz)."""
    stamp = lambda f: fmt_time(offset_s + f / fps)
    rallies, cur = [], None
    for h in hits:
        f = h.frame
        if cur is None or f - cur.shots[-1].frame > max_pause_s * fps:
            if cur is not None:
                rallies.append(cur)
            cur = Rally(len(rallies) + 1, f, start_time=stamp(f))
        if h.player is None:
            cur.shots.append(Shot(f, stamp(f), f"{h.team}-?", h.team,
                                  "unbekannt (Ball verdeckt)", ""))
            continue
        shot_type, hand = classify_shot(h.player, path[f], not cur.shots)
        cx, cy = h.player.court_pos
        cur.shots.append(Shot(f, stamp(f), h.player.name, h.team, shot_type, hand,
                              round(cx, 2), round(cy, 2), role=role_of(h.player.name)))
    if cur is not None:
        rallies.append(cur)

    for i, r in enumerate(rallies):
        last = r.shots[-1]
        limit = rallies[i + 1].start_frame if i + 1 < len(rallies) else max(path, default=0) + 1
        after = [f for f in path if last.frame < f < limit]
        r.end_frame = max(after, default=last.frame)
        if len(after) < 5:
            r.result = "Unklar (Ball verloren)"
            continue
        # Letzte Ballposition: am Ende liegt/rollt der Ball meist am Boden -> Homographie gilt
        side = Court.team_of(court.to_court(path[r.end_frame]))
        if side != last.team:
            r.result, r.winner_team = "Winner (Gegner erreicht den Ball nicht)", last.team
        else:
            r.result, r.winner_team = "Fehler: Netz", other(last.team)
        r.decided_by = last.player
    return rallies


# --------------------------------------------------------------------------
# Scoreboard: Teamnamen, Aufschlag-Stern und Spielstand per Texterkennung
# --------------------------------------------------------------------------
POINT_RANK = {"0": 0, "00": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


class Scoreboard:
    """Liest die TV-Einblendung "NAME / NAME  [*]  Saetze Spiele Punkte".
    Nutzt die macOS-Texterkennung (Vision, Paket ocrmac). Ohne ocrmac bleibt
    alles beim Alten (Teams heissen dann nur A und B)."""

    def __init__(self, region=(0, 0, 480, 120)):
        self.region = region
        try:
            from ocrmac import ocrmac
            self.ocr = ocrmac.OCR
        except ImportError:
            print("Hinweis: 'pip install ocrmac' fuer Namen vom Scoreboard (nur macOS).")
            self.ocr = None

    def read(self, frame) -> Optional[list]:
        """-> [{"name", "serving", "games", "points"}, {...}] (obere Zeile zuerst)"""
        if self.ocr is None:
            return None
        from PIL import Image
        x, y, w, h = self.region
        crop = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop).resize((w * 3, h * 3), Image.LANCZOS)
        lines = self.ocr(img, recognition_level="accurate").recognize()
        rows = []          # (y_mitte, name, serving)
        numbers = []       # (y_mitte, x, [tokens])
        for text, conf, (bx, by, bw, bh) in lines:
            yc = 1 - by - bh / 2              # Vision: Ursprung unten links
            if "/" in text:
                names, _, rest = text.partition("/")
                second = "".join(ch for ch in rest if ch.isalpha() or ch in " '-").strip()
                serving = any(not (ch.isalnum() or ch in " '-/.") for ch in rest)
                rows.append((yc, f"{names.strip().title()} / {second.title()}", serving))
            else:
                toks = [t for t in text.upper().split() if t in POINT_RANK or t.isdigit()]
                if toks:
                    numbers.append((yc, bx, toks))
        if len(rows) != 2:
            return None
        rows.sort()
        result = []
        for yc, name, serving in rows:
            toks = []
            for ny, nx, t in sorted(numbers, key=lambda n: n[1]):
                if abs(ny - yc) < 0.12:
                    toks += t
            if len(toks) < 2:
                return None
            result.append({"name": name, "serving": serving,
                           "games": int(toks[-2]) if toks[-2].isdigit() else 0,
                           "points": toks[-1]})
        return result


def _score_key(t):
    pts = t["points"]
    return t["games"], POINT_RANK.get(pts, int(pts) if pts.isdigit() else 0)


def stable_readings(readings):
    """Lesefehler aussortieren: ein Spielstand zaehlt nur, wenn er in zwei
    aufeinanderfolgenden Lesungen gleich ist (Einblendung wechselt, OCR erwischt
    manchmal die Satz- statt der Spielspalte)."""
    readings = [(f, r) for f, r in readings if r]
    keys = [tuple((t["name"], *_score_key(t)) for t in r) for _, r in readings]
    return [readings[i] for i in range(len(readings))
            if (i > 0 and keys[i] == keys[i - 1]) or (i + 1 < len(keys) and keys[i] == keys[i + 1])]


def legal_step(before, after):
    """Genau ein Punkt Unterschied nach Padel-Zaehlweise (inkl. Spiel- und Einstandsfall)."""
    if [t["name"] for t in before] != [t["name"] for t in after]:
        return False
    (g0a, p0a), (g0b, p0b) = map(_score_key, before)
    (g1a, p1a), (g1b, p1b) = map(_score_key, after)
    if (g1a, p1a, g1b, p1b) == (g0a, p0a, g0b, p0b):
        return False
    for (ga, pa, gb, pb, ha, qa, hb, qb) in ((g0a, p0a, g0b, p0b, g1a, p1a, g1b, p1b),
                                             (g0b, p0b, g0a, p0a, g1b, p1b, g1a, p1a)):
        if ha == ga and hb == gb and qb == pb and qa == pa + 1:
            return True                                   # Punkt im selben Spiel
        if ha == ga and hb == gb and pa == 3 and pb == 4 and qa == 3 and qb == 3:
            return True                                   # Einstand: AD des Gegners weg
        if ha == ga + 1 and hb == gb and qa == 0 and qb == 0:
            return True                                   # Spiel gewonnen
        if ha == 0 and hb == 0 and qa == 0 and qb == 0 and ga >= 5:
            return True                                   # Satz gewonnen
    return False


def score_text(reading):
    a, b = reading
    return f"{a['name']} {a['points']}:{b['points']} {b['name']} (Spiele {a['games']}:{b['games']})"


def player_name(team_name, role, revers):
    """Einzelname, wenn bekannt ist, wer Revés spielt (--revers), sonst Team + Position."""
    if not team_name:
        return ""
    names = [n.strip() for n in team_name.split("/")]
    rev = [n for n in names if n.lower() in revers]
    if role and len(names) == 2 and len(rev) == 1:
        return rev[0] if role == "Revés" else next(n for n in names if n != rev[0])
    return f"{team_name} ({role})" if role else team_name


def apply_scoreboard(rallies, readings, max_wait_frames, revers=(), lefties=()):
    """Teamnamen den Platzseiten zuordnen und Gewinner aus der Spielstandsaenderung
    bestimmen. readings: Liste (frame, reading) aufsteigend nach frame.
    Zuordnung: Das Team mit dem Stern schlaegt auf; auf welcher Seite der
    Aufschlag gespielt wurde, weiss die Analyse (erster Schlag des Ballwechsels).
    Aendert sich der Spielstand bis zum naechsten Ballwechsel nicht, war es kein
    Punkt (Fehlaufschlag mit zweitem Aufschlag, Let, Einspielen, Ball zuspielen)."""
    readings = stable_readings(readings)
    revers = {n.lower() for n in revers}
    lefties = {n.lower() for n in lefties}
    for i, r in enumerate(rallies):
        next_start = rallies[i + 1].start_frame if i + 1 < len(rallies) else 10 ** 12
        before = [rd for f, rd in readings if f <= r.start_frame]
        if not before:
            continue
        before = before[-1]
        # Der Stern fehlt kurz, waehrend die Einblendung wechselt -> letzte Lesung mit Stern
        starred = [rd for f, rd in readings if f <= r.start_frame and any(t["serving"] for t in rd)]
        server = next((t for t in starred[-1] if t["serving"]), None) if starred else None
        if server is not None and r.shots[0].shot_type == "Aufschlag":
            serve_side = r.shots[0].team
            receiver = next(t for t in starred[-1] if t is not server)
            r.team_names = {serve_side: server["name"], other(serve_side): receiver["name"]}
            for s in r.shots:
                s.team_name = r.team_names[s.team]
                s.player_name = player_name(s.team_name, s.role, revers)
                if s.player_name.lower() in lefties and s.hand in ("Vorhand", "Rueckhand"):
                    s.hand = "Rueckhand" if s.hand == "Vorhand" else "Vorhand"
        r.score_before = score_text(before)

        key = lambda t: (t["games"], POINT_RANK.get(t["points"], int(t["points"])
                                                    if t["points"].isdigit() else 0))
        limit = min(r.end_frame + max_wait_frames, next_start)
        window = [rd for f, rd in readings if r.end_frame < f <= limit]
        after = next((rd for rd in window if legal_step(before, rd)), None)
        if after is None:
            if window:
                r.result, r.winner_team, r.decided_by = "kein Punkt (Spielstand unveraendert)", "", ""
            continue
        r.score_after = score_text(after)
        gained = [key(a) > key(b) for a, b in zip(after, before)]
        lost = [key(a) < key(b) for a, b in zip(after, before)]
        if gained.count(True) == 1:
            winner = after[gained.index(True)]["name"]
        elif lost.count(True) == 1:            # Einstand: AD des Gegners faellt weg
            winner = after[1 - lost.index(True)]["name"]
        else:
            continue                           # Satzende: Spiele beider Teams auf 0
        r.winner_name = winner
        side = next((k for k, v in r.team_names.items() if v == winner), "")
        if side and side != r.winner_team:
            # Scoreboard ist die Wahrheit; Analyse hatte den Ausgang falsch gedeutet
            r.result, r.winner_team, r.decided_by = "Punkt laut Scoreboard", side, ""


def player_stats(rallies):
    """Zaehlt pro Spielerin (Name bzw. Team + Position), ohne Scoreboard pro Platzseite."""
    who = lambda s: s.player_name or s.player
    stats = defaultdict(lambda: defaultdict(int))
    for r in rallies:
        for s in r.shots:
            stats[who(s)]["Schlaege"] += 1
            stats[who(s)][s.shot_type] += 1
            if s.hand:
                stats[who(s)][s.hand] += 1
        if r.decided_by:
            key = "Winner" if r.result.startswith("Winner") else r.result
            last = next(s for s in reversed(r.shots) if s.player == r.decided_by)
            stats[who(last)][key] += 1
    return stats


# --------------------------------------------------------------------------
# Zeichnen
# --------------------------------------------------------------------------
def draw_frame(img, court, players, path, idx, message):
    court.draw(img)
    for p in players:
        x1, y1, x2, y2 = map(int, p.box)
        color = (255, 140, 0) if p.team == "A" else (0, 140, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, p.name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    trail = [path[f] for f in range(idx - 15, idx + 1) if f in path]
    for a, b in zip(trail, trail[1:]):
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
    if idx in path:
        cv2.circle(img, tuple(map(int, path[idx])), 6, (0, 0, 255), 2)

    if message:
        # OpenCV kann keine Akzente zeichnen
        text = message.replace("é", "e").replace("É", "E")
        y = img.shape[0] - 20
        cv2.rectangle(img, (10, y - 28), (20 + 12 * len(text), y + 8), (0, 0, 0), -1)
        cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def overlay_messages(rallies, fps, show_s=2.0):
    """Pro Frame die einzublendende Meldung (Schlag oder Punktende), als Liste
    (von_frame, bis_frame, text) aufsteigend."""
    events = []
    for r in rallies:
        for s in r.shots:
            events.append((s.frame, f"{s.player_name or s.player}: {s.shot_type}"))
        winner = r.team_names.get(r.winner_team) or (f"Team {r.winner_team}" if r.winner_team else "")
        events.append((r.end_frame, f"Punkt {r.id}: {r.result}" + (f" -> {winner}" if winner else "")))
    events.sort(key=lambda e: e[0])
    spans = []
    for i, (f, text) in enumerate(events):
        until = min(f + int(show_s * fps), events[i + 1][0] if i + 1 < len(events) else 10 ** 12)
        spans.append((f, until, text))
    return spans


# --------------------------------------------------------------------------
# Videoquelle: lokale Datei oder YouTube-Link
# --------------------------------------------------------------------------
def parse_time(s) -> Optional[float]:
    """'1:23:45', '83:00' oder '4980' -> Sekunden."""
    if not s:
        return None
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def fmt_time(sec) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"


def is_url(s) -> bool:
    return s.startswith(("http://", "https://", "www.", "youtu"))


def download_video(url, start=None, end=None, height=720, cache_dir="videos") -> str:
    """Laedt nur die Videospur (kein Ton) als H.264 bis `height` Pixel.
    Mit start/end wird nur der Ausschnitt geladen. Bereits geladene Dateien
    werden wiederverwendet, abgebrochene Downloads fortgesetzt."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func

    with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("live_status") == "is_live":
        raise SystemExit("Stream laeuft noch live - bitte nach dem Ende erneut versuchen.")
    duration = info.get("duration") or 0

    tag = f"{fmt_time(start or 0)}-{fmt_time(end) if end else 'ende'}".replace(":", "")
    path = os.path.join(cache_dir, f"{info['id']}_{tag}_{height}p.mp4")
    title = info.get("title", "")
    if os.path.exists(path):
        print(f"Video aus Cache: {path}")
        return path, title
    os.makedirs(cache_dir, exist_ok=True)

    opts = {
        # H.264 bevorzugen: liest OpenCV ueberall zuverlaessig (AV1/VP9 nicht immer)
        "format": (f"bv*[vcodec^=avc1][height<={height}]/bv*[height<={height}]"
                   f"/b[height<={height}]/b"),
        "outtmpl": path,
        "noplaylist": True,
        "continuedl": True,
        "retries": 10,
        "fragment_retries": 10,
        "external_downloader_args": {"ffmpeg_i": ["-loglevel", "error", "-nostats"]},
    }
    if start or end:
        opts["download_ranges"] = download_range_func(None, [(start or 0, end or duration)])
    span = (end or duration) - (start or 0)
    print(f"Lade '{info.get('title')}' ({fmt_time(span)} von {fmt_time(duration)}) "
          f"-> {path}")
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not os.path.exists(path):
        raise SystemExit("Download fehlgeschlagen.")
    return path, title


def youtube_start(url) -> Optional[float]:
    """Startzeit aus ?t=3864 / &t=1h4m24s eines YouTube-Links."""
    from urllib.parse import urlparse, parse_qs
    t = parse_qs(urlparse(url).query).get("t", [None])[0]
    if not t:
        return None
    if t.isdigit():
        return float(t)
    sec, num = 0.0, ""
    for ch in t:
        if ch.isdigit():
            num += ch
        elif ch in "hms" and num:
            sec += int(num) * {"h": 3600, "m": 60, "s": 1}[ch]
            num = ""
    return sec


class Progress:
    def __init__(self, label, total, fps, first_frame):
        self.label, self.total, self.fps, self.first = label, total, fps, first_frame
        self.t0 = time.time()

    def tick(self, idx):
        if idx and idx % 300 == 0:
            speed = idx / (time.time() - self.t0)
            eta = f", Rest ca. {fmt_time((self.total - idx) / speed)}" if self.total else ""
            print(f"  [{self.label}] {fmt_time((self.first + idx) / self.fps)} im Video, "
                  f"{100 * idx / max(self.total, 1):.0f}%, {speed:.1f} fps{eta}")


def render_video(cap, first, first_frame, fps, court, players_per_frame, path, rallies,
                 out_dir, show=False):
    """Annotiertes analyse.mp4 schreiben (3. Durchgang)."""
    h, w = first.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    video_out = os.path.join(out_dir, "analyse.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    spans, k = overlay_messages(rallies, fps), 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    prog = Progress("Video", len(players_per_frame), fps, first_frame)
    for idx in range(len(players_per_frame)):
        ok, frame = cap.read()
        if not ok:
            break
        while k < len(spans) and spans[k][1] <= idx:
            k += 1
        message = spans[k][2] if k < len(spans) and spans[k][0] <= idx else None
        draw_frame(frame, court, players_per_frame[idx], path, idx, message)
        writer.write(frame)
        if show:
            cv2.imshow("Padel-Analyse", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        prog.tick(idx)
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    to_browser_video(video_out)


def small_gray(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180),
                      interpolation=cv2.INTER_AREA).astype(np.int16)


def is_duplicate(g, prev, threshold=3.5):
    """25->30-fps-Streams wiederholen jedes 6. Bild. Kopien unterscheiden sich nur durch
    gleichmaessiges Kompressionsrauschen (Block-Maximum <= 2.5), echte Bilder durch
    lokale Bewegung (>= 4.8)."""
    if prev is None:
        return False
    blocks = cv2.resize(np.abs(g - prev).astype(np.float32), (40, 23), interpolation=cv2.INTER_AREA)
    return float(blocks.max()) < threshold


def detect_pass(cap, first_frame, start_idx, n_frames, fps, players_tr, ball_det, scoreboard,
                sb_every, players_per_frame, peaks, sb_readings, player_every, skip_pauses,
                prog, save_cache, probe_s=0.5, rewind_s=1.5, idle_after_s=4.0,
                wake_prob=0.5, keep_prob=0.3):
    """1. Durchgang: Spieler und Ballkandidaten pro Frame (Listen werden erweitert).
    Mit skip_pauses wird zwischen den Ballwechseln nur alle probe_s Sekunden eine
    Stichprobe genommen; taucht der Ball auf, geht es rewind_s zurueck und ab dort wird
    jedes Frame analysiert, bis idle_after_s lang kein Ball mehr zu sehen ist.
    Rueckgabe: Anzahl voll analysierter Frames (None ohne skip_pauses)."""
    probe, rewind = max(int(probe_s * fps), 3), int(rewind_s * fps)
    idle_after, save_every = int(idle_after_s * fps), int(300 * fps)
    active = not skip_pauses
    idx, last_ball, prev_g, players, n_active, since_yolo = start_idx, start_idx, None, [], 0, 0
    probe_frames, active_end = [], start_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + idx)
    while idx < n_frames:
        need = (active or idx % sb_every == 0 or idx % probe >= probe - 3)
        if need:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        if not ok:
            break
        if idx % sb_every == 0:
            sb_readings.append((idx, scoreboard.read(frame)))

        if not active:
            players_per_frame.append([])
            peaks.append([])
            if idx % probe >= probe - 3:
                probe_frames.append(frame)
            if idx % probe == probe - 1:
                ball_det.reset()
                found = []
                for f in probe_frames[-3:]:
                    found = ball_det.peaks(f)
                probe_frames = []
                if found and found[0][0] >= wake_prob:
                    # Ball gesehen: zurueckspringen, damit der Anfang (Aufschlag) dabei ist
                    target = max(idx - rewind, active_end, start_idx)
                    del players_per_frame[target:], peaks[target:]
                    idx, active, last_ball, prev_g = target, True, idx, None
                    ball_det.reset()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + idx)
                    continue
        else:
            n_active += 1
            g = small_gray(frame)
            if is_duplicate(g, prev_g):
                # Kopie des vorigen Bildes: Ergebnis uebernehmen, Netze nicht fuettern
                players_per_frame.append(players_per_frame[-1] if players_per_frame else [])
                peaks.append(peaks[-1] if peaks else [])
            else:
                if since_yolo % player_every == 0:
                    players = players_tr.update(frame)
                since_yolo += 1
                players_per_frame.append(players)
                found = ball_det.peaks(frame)
                peaks.append(found)
                if found and found[0][0] >= keep_prob:
                    last_ball = idx
            prev_g = g
            if skip_pauses and idx - last_ball > idle_after:
                active, active_end, prev_g, probe_frames = False, idx + 1, None, []
        prog.tick(idx - start_idx)
        idx += 1
        if idx % save_every == 0:
            save_cache()
    return n_active if skip_pauses else None


# --------------------------------------------------------------------------
# Ausgabe: browsertaugliches Video und Dashboard
# --------------------------------------------------------------------------
def to_browser_video(path):
    """OpenCV schreibt MPEG-4 Part 2, das Browser nicht abspielen -> H.264 via ffmpeg."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        print("Hinweis: ohne ffmpeg spielt der Browser analyse.mp4 nicht ab (brew install ffmpeg).")
        return
    tmp = path + ".tmp.mp4"
    os.replace(path, tmp)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", path], check=True)
    os.remove(tmp)


def write_dashboard(out_dir, title, source, offset_s, fps, rallies, stats,
                    length_s, original_video=None):
    """dashboard.html aus der Vorlage neben padel.py; Daten werden eingebettet.
    Video: analyse.mp4 im selben Ordner, oder (ohne Rendern) das Originalvideo,
    relativ verlinkt und ab offset_s abgespielt."""
    template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "server", "templates", "dashboard.html")
    data = {
        "title": title,
        "source": source,
        "offset": offset_s,
        "length": length_s,
        "video_src": (os.path.relpath(original_video, out_dir).replace(os.sep, "/")
                      if original_video else "analyse.mp4"),
        "video_offset": offset_s if original_video else 0,
        "fps": fps,
        "teams": list(dict.fromkeys(n for r in rallies for n in
                                    sorted(r.team_names.values()))),
        "rallies": [asdict(r) for r in rallies],
        "stats": {name: dict(v) for name, v in stats.items()},
    }
    with open(template, encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", payload)
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Padel-Analyse: Punkte, Schlaege, Fehler")
    ap.add_argument("--video", required=True, help="Videodatei oder YouTube-Link")
    ap.add_argument("--start", help="Startzeit, z.B. 1:05:00 (Standard: ?t= im Link)")
    ap.add_argument("--end", help="Endzeit, z.B. 1:20:00")
    ap.add_argument("--height", type=int, default=720,
                    help="max. Aufloesung beim Download (Standard 720)")
    ap.add_argument("--court", help="court.json mit 4 Ecken")
    ap.add_argument("--calibrate", action="store_true", help="Ecken interaktiv anklicken")
    ap.add_argument("--player-model", default="models/yolov8s.pt")
    ap.add_argument("--ball-model", default="models/ball_tracknet.pt",
                    help="TrackNet-Gewichte (werden beim ersten Lauf geladen)")
    ap.add_argument("--revers", default="",
                    help="Wer auf der Revés-Seite (links) spielt, z.B. 'Navarro,Lobo' "
                         "-> Statistik mit Einzelnamen")
    ap.add_argument("--lefties", default="", help="Linkshaenderinnen, z.B. 'Navarro'")
    ap.add_argument("--no-video", action="store_true",
                    help="kein analyse.mp4 rendern; das Dashboard spielt das Originalvideo")
    ap.add_argument("--no-skip", action="store_true",
                    help="auch Pausen zwischen den Ballwechseln voll analysieren (langsamer)")
    ap.add_argument("--player-every", type=int, default=2,
                    help="Spielererkennung nur jeden n-ten Frame (schneller)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out-dir", default="result",
                    help="Ordner fuer analyse.mp4, events.json, stats.csv")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    start, end = parse_time(args.start), parse_time(args.end)
    # Messwerte fuer die Zeitschaetzung kuenftiger Analysen (timing.json)
    timing, t_mark = {}, time.time()
    if is_url(args.video):
        if start is None:
            start = youtube_start(args.video)
        # ganzes Video einmal laden (schneller als Ausschnitte), dann lokal springen
        video, title = download_video(args.video, height=args.height)
        timing["download_s"] = round(time.time() - t_mark, 1)
    else:
        video, title = args.video, os.path.basename(args.video)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    first_frame = int((start or 0) * fps)
    last_frame = int(end * fps) if end else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Leeres Video (oder --start hinter dem Ende).")

    if args.calibrate or not args.court:
        court = calibrate_interactive(first)
        os.makedirs("courts", exist_ok=True)
        court.save("courts/court.json")
        print("Kalibrierung in courts/court.json gespeichert.")
    else:
        court = Court.load(args.court)
    n_frames = max(last_frame - first_frame, 0)
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames) if n_frames else args.max_frames
    print(f"Analysiere {fmt_time(first_frame / fps)} - "
          f"{fmt_time((first_frame + n_frames) / fps)} ({n_frames} Frames)")

    # ---- 1. Durchgang: Spieler + Ballkandidaten -----------------------------
    t_mark = time.time()
    players_tr = PlayerTracker(court, args.player_model)
    ball_det = BallDetector(args.ball_model)
    split_names = lambda v: [n.strip() for n in v.split(",") if n.strip()]
    scoreboard = Scoreboard()
    timing["load_s"] = round(time.time() - t_mark, 1)
    sb_every = int(2 * fps)
    players_per_frame, peaks, sb_readings = [], [], []
    # Zwischenspeicher: lange Laeufe koennen nach einem Abbruch weitermachen
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, f".erkennung_{first_frame}_{n_frames}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            players_per_frame, peaks, sb_readings = pickle.load(f)
        print(f"Zwischenstand geladen: {fmt_time(len(peaks) / fps)} bereits erkannt.")

    def save_cache():
        with open(cache + ".tmp", "wb") as f:
            pickle.dump((players_per_frame, peaks, sb_readings), f)
        os.replace(cache + ".tmp", cache)

    start_idx = len(peaks)
    prog = Progress("Erkennung", n_frames, fps, first_frame)
    t_mark = time.time()
    stats_active = detect_pass(cap, first_frame, start_idx, n_frames, fps, players_tr, ball_det,
                               scoreboard, sb_every, players_per_frame, peaks, sb_readings,
                               args.player_every, not args.no_skip, prog, save_cache)
    timing.update(detect_s=round(time.time() - t_mark, 1),
                  processed_s=round((len(peaks) - start_idx) / fps, 1),
                  active_s=round((stats_active if stats_active is not None
                                  else len(peaks) - start_idx) / fps, 1))
    save_cache()
    if stats_active is not None:
        print(f"Voll analysiert: {fmt_time(stats_active / fps)} von "
              f"{fmt_time((n_frames - start_idx) / fps)} (Rest waren Pausen).")
    save_cache()
    # Der neue Spielstand erscheint erst ein paar Sekunden nach dem letzten Punkt
    for extra in range(len(peaks) + sb_every, len(peaks) + int(20 * fps), sb_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + extra)
        ok, frame = cap.read()
        if not ok:
            break
        sb_readings.append((extra, scoreboard.read(frame)))

    # ---- 2. Durchgang: Ballbahn, Schlaege, Ballwechsel ----------------------
    t_mark = time.time()
    path = build_ball_path(peaks)
    hits = detect_hits(path, players_per_frame, fps)
    rallies = build_rallies(hits, path, court, fps, first_frame / fps)
    apply_scoreboard(rallies, sb_readings, max_wait_frames=int(25 * fps),
                     revers=split_names(args.revers), lefties=split_names(args.lefties))
    print(f"Ball in {len(path)} von {len(peaks)} Frames gefunden, {len(hits)} Schlaege.")
    verbose = len(rallies) <= 5
    for r in rallies:
        if verbose and r.score_before:
            print(f"  Stand: {r.score_before}")
        if verbose:
            for s in r.shots:
                print(f"  [{s.time}] {s.player_name or s.player}: {s.shot_type} ({s.hand})")
        winner = r.team_names.get(r.winner_team) or (f"Team {r.winner_team}" if r.winner_team else "")
        print(f"  Punkt {r.id} [{r.start_time}, {len(r.shots)} Schlaege]: {r.result}"
              + (f" -> {winner}" if winner else ""))
        if verbose and r.score_after:
            print(f"  Neuer Stand: {r.score_after}")

    timing["eval_s"] = round(time.time() - t_mark, 1)
    # ---- 3. Durchgang: Video schreiben (abschaltbar mit --no-video) ---------
    t_mark = time.time()
    if args.no_video:
        cap.release()
    else:
        render_video(cap, first, first_frame, fps, court, players_per_frame, path, rallies,
                     args.out_dir, args.show)
        timing["render_s"] = round(time.time() - t_mark, 1)

    with open(os.path.join(args.out_dir, "events.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rallies], f, indent=2, ensure_ascii=False)

    stats = player_stats(rallies)
    cols = sorted({k for s in stats.values() for k in s})
    with open(os.path.join(args.out_dir, "stats.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Spieler"] + cols)
        for name in sorted(stats):
            wr.writerow([name] + [stats[name].get(c, 0) for c in cols])

    print(f"\n{len(rallies)} Ballwechsel analysiert.")
    for name in sorted(stats):
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats[name].items())))
    write_dashboard(args.out_dir, title, args.video, first_frame / fps, fps, rallies, stats,
                    len(peaks) / fps, original_video=video if args.no_video else None)
    timing.update(range_s=round(len(peaks) / fps, 1), player_every=args.player_every,
                  skip_pauses=not args.no_skip, render_video=not args.no_video,
                  coreml=coreml_available(), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(args.out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"Gespeichert in {args.out_dir}/: " + ("" if args.no_video else "analyse.mp4, ")
          + "events.json, stats.csv, dashboard.html")


if __name__ == "__main__":
    main()
