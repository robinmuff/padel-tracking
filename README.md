# Padel-Tracking

Analysiert Padel-Videos (z. B. YouTube-Livestreams eines Courts): erkennt Ballwechsel,
Schläge pro Spielerin und den Gewinner jedes Punkts, und zeigt alles in einem
Dashboard mit Video, klickbarer Schlagliste und Platzgrafik des Spielzugs.

Gedacht für eine feste Kamera, erhöht hinter einer Grundlinie (typische Court-Kamera).

## Installation

Python 3.11 oder neuer, `ffmpeg` für YouTube-Ausschnitte und browsertaugliche Videos.

```sh
brew install ffmpeg python@3.13
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Die Modelle (YOLOv8s für Spielerinnen, TrackNet für den Ball) werden beim ersten Lauf
nach `models/` geladen und auf dem Mac einmalig für die Neural Engine umgewandelt.

## Bedienung im Browser

```sh
server/start.sh          # öffnet http://127.0.0.1:7984
```

- **Neue Analyse:** YouTube-Link (gern mit `?t=`) oder Videodatei, Start, Ende,
  Platz-Kalibrierung und Einstellungen (Video rendern, Pausen überspringen, …).
- **Analysen:** Fortschritt, Stoppen, Fortsetzen (Zwischenstand alle 5 Spielminuten),
  Neu auswerten, Dashboard.
- **Platz kalibrieren:** die vier Platzecken auf einem Videobild anklicken
  (einmal pro Kamera/Platz, gespeichert in `courts/`).

## Kommandozeile

```sh
.venv/bin/python padel.py --video "https://www.youtube.com/live/<id>?t=3864" \
    --end 1:04:43 --court courts/milano-court2.json --no-video
```

Wichtige Optionen: `--start/--end`, `--no-video` (kein annotiertes Video rendern),
`--no-skip` (Pausen nicht überspringen), `--revers "Name,Name"` (wer links/Revés spielt →
Statistik mit Einzelnamen), `--lefties`, `--out-dir`.

Ergebnisse pro Analyse: `dashboard.html`, `events.json`, `stats.csv`, optional `analyse.mp4`.

## So funktioniert es

1. **Erkennung:** Spielerinnen mit YOLO + ByteTrack, Ball mit TrackNet (3 aufeinander­folgende
   Bilder → Ball-Wahrscheinlichkeit pro Pixel). Zwischen den Ballwechseln werden nur
   Stichproben genommen; doppelte Bilder (25→30-fps-Streams) werden übersprungen.
2. **Auswertung:** Ballkandidaten werden zu Flugbahnen verkettet; ein Schlag ist ein
   sprunghafter Richtungswechsel des Balls bei einer Spielerin. Teamnamen, Aufschlag und
   Spielstand liest die Texterkennung vom TV-Scoreboard (macOS); der Spielstand bestätigt
   den Gewinner jedes Punkts.
3. **Dashboard:** Video (gerendert oder Original) mit Sprung zu jedem Schlag.

## Grenzen

- Schlaganzahl und Schlagart sind Näherungen; verdeckte Schläge werden erschlossen
  („unbekannt (Ball verdeckt)“).
- Wer im Team links bzw. rechts spielt, verrät das Scoreboard nicht → `--revers`.
- Texterkennung des Scoreboards und Neural Engine nur auf macOS; sonst läuft alles mit
  PyTorch, aber ohne Teamnamen.

## Herkunft der Ball-Gewichte

Die TrackNet-Gewichte werden beim ersten Lauf aus dem Release des Projekts
[Padex](https://github.com/rondo-labs/Padex) geladen und nicht mit diesem Repository
verteilt. Bitte deren Lizenzbedingungen beachten.
