"""Missingness injection and target selection.

Each mask removes round(p*E) cells from the same E eligible cells. The mechanisms
differ only in the sampling weights:

MCAR  equal weights.
MAR   sigmoid(2*z) of the row's standardized driver value. Categorical drivers are
      frequency-rank encoded first. The driver column is never masked.
MNAR  numeric cells: sigmoid(2*z) of the cell's own standardized value. Categorical
      cells: weight 3 if the value is in a seeded random half of the domain, else 1.
      Weights are rescaled to mean 1 within each column before sampling.

Sampling is weighted, without replacement. Each mask then gets min(200, N) targets,
split over columns in proportion (largest remainder), seed = mask seed + 10000.
"""
from __future__ import annotations

import hashlib
import io
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Table

MECHANISMS = ("MCAR", "MAR", "MNAR")
MECH_CODE = {"MCAR": 1, "MAR": 2, "MNAR": 3}
TARGETS_PER_MASK = 200
TARGET_SEED_OFFSET = 10_000
FLAG_BELOW = 10
MNAR_CATEGORICAL_WEIGHT = 3.0


def _table_code(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def _entropy(table: str, mechanism: str, rate: float, seed: int) -> list[int]:
    return [int(seed), MECH_CODE[mechanism], int(round(rate * 100)), _table_code(table)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / sd


def frequency_rank(values: pd.Series) -> np.ndarray:
    counts = values.value_counts()
    order = sorted(counts.index, key=lambda v: (-counts[v], str(v)))
    rank = {v: i + 1 for i, v in enumerate(order)}
    return values.map(rank).to_numpy(dtype=float)


def driver_values(values: pd.Series, transform: str | None) -> pd.Series:
    if transform is None:
        return values
    if transform == "area_code":
        codes = values.astype(str).str.extract(r"^\D*(\d{3})")[0]
        if codes.isna().any():
            raise ValueError("area_code transform: a phone value has no leading 3-digit code")
        return codes
    raise ValueError(f"unknown MAR driver transform {transform!r}")


def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


MNAR_RULES = ("frozen", "per_column_normalized", "per_column_stratified")


def mechanism_weights(table: Table, cells: pd.DataFrame, mechanism: str,
                      rng: np.random.Generator, mnar_rule: str = "frozen") -> np.ndarray:
    if mechanism == "MCAR":
        return np.ones(len(cells))
    if mechanism == "MAR":
        if table.mar_driver is None:
            raise ValueError(f"{table.name}: MAR needs a driver column")
        d = table.clean[table.mar_driver]
        if d.isna().any():
            raise ValueError(f"{table.name}: MAR driver {table.mar_driver!r} has missing values")
        d = driver_values(d, table.mar_driver_transform)
        x = d.to_numpy(dtype=float) if table.types[table.mar_driver] == "numeric" else frequency_rank(d)
        z = _standardize(x)
        return _sigmoid(2.0 * z[cells["row"].to_numpy()])
    if mechanism == "MNAR":
        w = np.ones(len(cells))
        for c in table.targets:
            idx = np.flatnonzero(cells["column"].to_numpy() == c)
            rows = cells["row"].to_numpy()[idx]
            vals = table.clean[c]
            if table.types[c] == "numeric":
                z = _standardize(vals.to_numpy(dtype=float))
                w[idx] = _sigmoid(2.0 * z[rows])
            else:
                domain = sorted(vals.dropna().unique(), key=str)
                half = rng.choice(len(domain), size=round_half_up(len(domain) / 2), replace=False)
                heavy = {domain[i] for i in half}
                w[idx] = [MNAR_CATEGORICAL_WEIGHT if v in heavy else 1.0 for v in vals.to_numpy()[rows]]
            if mnar_rule != "frozen" and len(idx):
                w[idx] = w[idx] / w[idx].mean()
        return w
    raise ValueError(f"unknown mechanism {mechanism!r}")


def make_mask(table: Table, mechanism: str, rate: float, seed: int, mnar_rule: str = "frozen") -> pd.DataFrame:
    if mnar_rule not in MNAR_RULES:
        raise ValueError(f"unknown MNAR rule {mnar_rule!r}")
    cells = table.eligible_cells()
    ss = np.random.SeedSequence(_entropy(table.name, mechanism, rate, seed))
    weight_rng, sample_rng = (np.random.default_rng(s) for s in ss.spawn(2))
    w = mechanism_weights(table, cells, mechanism, weight_rng, mnar_rule=mnar_rule if mechanism == "MNAR" else "frozen")
    k = round_half_up(rate * len(cells))
    if mechanism == "MNAR" and mnar_rule == "per_column_stratified":
        col = cells["column"].to_numpy()
        counts = {c: int((col == c).sum()) for c in table.targets}
        alloc = largest_remainder(counts, k, sample_rng)
        parts = []
        for c in table.targets:
            idx = np.flatnonzero(col == c)
            if alloc[c]:
                wc = w[idx]
                parts.append(sample_rng.choice(idx, size=alloc[c], replace=False, p=wc / wc.sum()))
        pick = np.concatenate(parts) if parts else np.array([], dtype=int)
    else:
        pick = sample_rng.choice(len(cells), size=k, replace=False, p=w / w.sum())
    out = cells.iloc[np.sort(pick)].reset_index(drop=True)
    out["row"] = out["row"].astype("int32")
    return out


def largest_remainder(counts: dict[str, int], n: int, rng: np.random.Generator) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        return {c: 0 for c in counts}
    exact = {c: n * k / total for c, k in counts.items()}
    alloc = {c: int(np.floor(v)) for c, v in exact.items()}
    left = n - sum(alloc.values())
    cols = list(counts)
    tiebreak = {c: r for c, r in zip(cols, rng.permutation(len(cols)))}
    order = sorted(cols, key=lambda c: (-(exact[c] - alloc[c]), tiebreak[c]))
    for c in order[:left]:
        alloc[c] += 1
    return alloc


@dataclass
class TargetSelection:
    targets: pd.DataFrame
    per_column: pd.DataFrame


def select_targets(table: Table, mask: pd.DataFrame, mechanism: str, rate: float, seed: int,
                   per_mask: int = TARGETS_PER_MASK) -> TargetSelection:
    rng = np.random.default_rng(np.random.SeedSequence(
        _entropy(table.name, mechanism, rate, seed + TARGET_SEED_OFFSET)))
    N = len(mask)
    n = min(per_mask, N)
    counts = {c: int((mask["column"] == c).sum()) for c in table.targets}
    alloc = largest_remainder(counts, n, rng)
    parts = []
    for c in table.targets:
        rows = np.sort(mask.loc[mask["column"] == c, "row"].to_numpy())
        if alloc[c]:
            chosen = np.sort(rng.choice(rows, size=alloc[c], replace=False))
            parts.append(pd.DataFrame({"row": chosen.astype("int32"), "column": c}))
    targets = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"row": [], "column": []})
    rows_pc = []
    for c in table.targets:
        Nc, nc = counts[c], alloc[c]
        weight = (Nc / N) / (nc / n) if nc else None
        rows_pc.append({"column": c, "N_c": Nc, "n_c": nc,
                        "inclusion_prob": (nc / Nc) if Nc else None,
                        "mask_share": Nc / N if N else None, "target_share": nc / n if n else None,
                        "weight": weight, "flagged_lt_10": 0 < nc < FLAG_BELOW or (Nc > 0 and nc == 0)})
    return TargetSelection(targets=targets, per_column=pd.DataFrame(rows_pc))


def content_sha256(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.sort_values(["column", "row"]).to_csv(buf, index=False, lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def mask_name(mechanism: str, rate: float, seed: int) -> str:
    return f"{mechanism}_{int(round(rate * 100))}_s{seed}"


def mask_report(table: Table, mask: pd.DataFrame, sel: TargetSelection,
                mechanism: str, rate: float, seed: int) -> dict:
    E = len(table.eligible_cells())
    per_col = []
    for c in table.targets:
        e_c = int(table.observed(c).sum())
        m_c = int((mask["column"] == c).sum())
        r = sel.per_column.set_index("column").loc[c]
        per_col.append({"column": c, "type": table.types[c], "eligible": e_c, "masked": m_c,
                        "realized_rate": round(m_c / e_c, 6) if e_c else None,
                        "targets": int(r["n_c"]),
                        "inclusion_prob": None if pd.isna(r["inclusion_prob"]) else round(float(r["inclusion_prob"]), 6),
                        "weight": None if pd.isna(r["weight"]) else round(float(r["weight"]), 6),
                        "flagged": bool(r["flagged_lt_10"])})
    eff = None
    if table.cluster_column is not None and len(sel.targets):
        labels = table.clean[table.cluster_column].to_numpy()[sel.targets["row"].to_numpy()]
        eff = int(pd.Series(labels).nunique())
    return {"name": mask_name(mechanism, rate, seed), "table": table.name, "mechanism": mechanism,
            "rate": rate, "seed": seed, "target_seed": seed + TARGET_SEED_OFFSET,
            "rng_entropy": _entropy(table.name, mechanism, rate, seed),
            "eligible_cells": E, "masked_cells": len(mask), "overall_rate": round(len(mask) / E, 6),
            "targets": len(sel.targets), "effective_clusters": eff, "cluster_column": table.cluster_column,
            "mask_content_sha256": content_sha256(mask),
            "targets_content_sha256": content_sha256(sel.targets), "columns": per_col}
