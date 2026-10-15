#!/bin/sh
# Padel-Analyse Server starten: http://127.0.0.1:7984
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python server/app.py "$@"
