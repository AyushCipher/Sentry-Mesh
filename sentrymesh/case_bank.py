"""Loader for the seeded synthetic case bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from sentrymesh.config import CASE_BANK_DIR
from sentrymesh.models import Case


def case_files(directory: Optional[Path] = None) -> List[Path]:
    root = directory or CASE_BANK_DIR
    return sorted(p for p in root.glob("*.json") if not p.name.startswith("_"))


def load_case_file(path: Path) -> Case:
    with path.open("r", encoding="utf-8") as handle:
        return Case.from_dict(json.load(handle))


def load_cases(directory: Optional[Path] = None) -> List[Case]:
    return [load_case_file(p) for p in case_files(directory)]


def load_case(case_id: str, directory: Optional[Path] = None) -> Optional[Case]:
    root = directory or CASE_BANK_DIR
    path = root / f"{case_id}.json"
    return load_case_file(path) if path.exists() else None


def raw_case(case_id: str, directory: Optional[Path] = None) -> Optional[Dict]:
    """The full case file including ground-truth labels - for the eval harness
    and the dashboard's eval pages only, never for an agent prompt."""
    root = directory or CASE_BANK_DIR
    path = root / f"{case_id}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
