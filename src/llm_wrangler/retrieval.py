from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from .data import Table


def format_value(v, numeric: bool) -> str:
    if numeric:
        f = float(v)
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return f"{f:.6g}"
    return str(v)


class MaskedTable:

    def __init__(self, table: Table, mask: pd.DataFrame):
        self.table = table
        self.columns = [c for c in table.clean.columns if table.types[c] != "id"]
        self.numeric = {c: table.types[c] == "numeric" for c in self.columns}
        masked: dict[str, np.ndarray] = {c: g["row"].to_numpy() for c, g in mask.groupby("column")}
        self.masked_cells = {(int(r), c) for c, rows in masked.items() for r in rows}
        disp: dict[str, list] = {}
        raw: dict[str, np.ndarray] = {}
        for c in self.columns:
            s = table.clean[c].to_numpy(dtype=object).copy()
            if c in masked:
                s[masked[c]] = None
            raw[c] = s
            disp[c] = [None if (x is None or (isinstance(x, float) and np.isnan(x))) else
                       format_value(x, self.numeric[c]) for x in s]
        self.display = pd.DataFrame({c: pd.Series(v, dtype=object) for c, v in disp.items()})
        self.values = raw
        self.n_rows = len(self.display)

    def observed(self, column: str) -> np.ndarray:
        return self.display[column].notna().to_numpy()

    def observed_values(self, column: str) -> np.ndarray:
        v = self.values[column]
        keep = self.observed(column)
        return v[keep]

    def row_items(self, row: int, columns: list[str] | None = None, exclude: set[str] = frozenset()) -> list[tuple[str, str]]:
        cols = columns if columns is not None else self.columns
        out = []
        for c in cols:
            if c in exclude:
                continue
            v = self.display.at[row, c]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out.append((c, v))
        return out

    @cached_property
    def domains(self) -> dict[str, list[str]]:
        out = {}
        for c in self.columns:
            if self.numeric[c]:
                continue
            vc = self.display[c].dropna().value_counts()
            out[c] = sorted(vc.index, key=lambda v: (-vc[v], str(v)))
        return out

    @cached_property
    def numeric_stats(self) -> dict[str, dict]:
        out = {}
        for c in self.columns:
            if self.numeric[c]:
                v = self.observed_values(c).astype(float)
                if len(v):
                    out[c] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
        return out


class Retriever:

    def __init__(self, mt: MaskedTable):
        self.mt = mt
        self._vec: dict[tuple, object] = {}

    def _matrix(self, exclude: tuple[str, ...]):
        if exclude not in self._vec:
            from sklearn.feature_extraction.text import TfidfVectorizer
            docs = [". ".join(f"{c}: {v}" for c, v in self.mt.row_items(r, exclude=set(exclude)))
                    for r in range(self.mt.n_rows)]
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=True, dtype=np.float64)
            self._vec[exclude] = vec.fit_transform(docs)
        return self._vec[exclude]

    def candidate_pool(self, row: int, columns: list[str]) -> np.ndarray:
        obs = np.stack([self.mt.observed(c) for c in columns])
        pool = np.flatnonzero(obs.all(axis=0))
        if len(pool) == 0:
            pool = np.flatnonzero(obs.any(axis=0))
        return pool[pool != row]

    def similar(self, row: int, columns: list[str], k: int) -> list[int]:
        if k <= 0:
            return []
        pool = self.candidate_pool(row, columns)
        if len(pool) == 0:
            return []
        X = self._matrix(tuple(sorted(columns)))
        sims = (X[pool] @ X[row].T).toarray().ravel()
        order = np.lexsort((pool, -sims))[:k]
        return [int(pool[i]) for i in order]

    def random(self, row: int, columns: list[str], k: int, rng: np.random.Generator) -> list[int]:
        if k <= 0:
            return []
        pool = self.candidate_pool(row, columns)
        if len(pool) == 0:
            return []
        pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [int(pool[i]) for i in pick]
