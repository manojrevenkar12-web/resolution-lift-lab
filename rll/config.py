"""Settings live in config/. The world reads world.yml and topics.yml; methods read methods.yml.
Neither side reads the other's file (tests/test_boundaries.py)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


def load_world() -> dict:
    return _load("world.yml")


def load_methods() -> dict:
    return _load("methods.yml")


def load_gates() -> dict:
    return _load("gates.yml")


def load_seeds() -> dict:
    return _load("seeds.yml")
