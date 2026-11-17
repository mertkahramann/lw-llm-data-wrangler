from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASETS_YAML = ROOT / "configs" / "datasets.yaml"

NEAR_UNIQUE_THRESHOLD = 0.85


@dataclass
class Table:
    name: str
    clean: pd.DataFrame
    types: dict[str, str]
    targets: list[str]
    excluded: dict[str, str] = field(default_factory=dict)
    mar_driver: str | None = None
    cluster_column: str | None = None
    mar_driver_transform: str | None = None

    @property
    def numeric_targets(self) -> list[str]:
        return [c for c in self.targets if self.types[c] == "numeric"]

    @property
    def categorical_targets(self) -> list[str]:
        return [c for c in self.targets if self.types[c] != "numeric"]

    def observed(self, column: str) -> np.ndarray:
        return self.clean[column].notna().to_numpy()

    def eligible_cells(self) -> pd.DataFrame:
        parts = []
        for c in self.targets:
            rows = np.flatnonzero(self.observed(c))
            parts.append(pd.DataFrame({"row": rows, "column": c}))
        return pd.concat(parts, ignore_index=True)


def load_config(name: str) -> dict:
    with DATASETS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)[name]


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_numeric(value: str | None, kind: str) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    v = value.strip()
    if kind == "percent":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", v)
    elif kind == "count_patients":
        m = re.fullmatch(r"(\d+)\s+patients?", v)
    else:
        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", v)
    if not m:
        raise ValueError(f"cannot parse {value!r} as {kind}")
    return float(m.group(1))


def load_table(name: str, csv_path: str | Path, config: dict | None = None) -> Table:
    cfg = config if config is not None else load_config(name)
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    markers = set(cfg.get("missing_markers", [""]))
    numeric = cfg.get("numeric_columns", {}) or {}
    ids = set(cfg.get("id_columns", []))
    text = set(cfg.get("text_columns", []))
    driver = cfg.get("mar_driver")

    data: dict[str, pd.Series] = {}
    types: dict[str, str] = {}
    for c in raw.columns:
        values = [None if x in markers else x for x in raw[c].tolist()]
        if c in numeric:
            types[c] = "numeric"
            data[c] = pd.Series([parse_numeric(x, numeric[c]) for x in values], dtype=float)
        else:
            types[c] = "id" if c in ids else ("text" if c in text else "categorical")
            data[c] = pd.Series(values, dtype=object)
    clean = pd.DataFrame(data)

    targets: list[str] = []
    excluded: dict[str, str] = {}
    for c in clean.columns:
        obs = clean[c].dropna()
        if types[c] == "id":
            excluded[c] = "(a) row identifier"
        elif obs.nunique() < 2:
            excluded[c] = "(b) constant (<2 distinct observed values)"
        elif types[c] != "numeric" and obs.nunique() / max(len(obs), 1) >= NEAR_UNIQUE_THRESHOLD:
            excluded[c] = f"(c) near-unique ({obs.nunique() / len(obs):.2f})"
        elif c == driver:
            excluded[c] = "(d) MAR driver (observed under all mechanisms)"
        else:
            targets.append(c)
    if driver is not None and driver not in excluded:
        raise ValueError(f"MAR driver {driver!r} was not excluded")
    return Table(name=name, clean=clean, types=types, targets=targets, excluded=excluded,
                 mar_driver=driver, cluster_column=cfg.get("cluster_column"),
                 mar_driver_transform=cfg.get("mar_driver_transform"))
