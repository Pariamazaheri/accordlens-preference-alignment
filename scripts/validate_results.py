from pathlib import Path

from accordlens.audit import validate

validate(Path(__file__).resolve().parents[1])
