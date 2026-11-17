from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .llm import RAW_ROLE, RAW_STOP
from .retrieval import MaskedTable, format_value

MAX_CANDIDATES = 20
P4_MAX_CELLS = 5

SYSTEM_INSTRUCT = ("You fill in missing values in rows of a data table. Reply with the missing value only, "
                   "on a single line, without any explanation. If the value cannot be inferred, reply: unknown.")
SYSTEM_BATCH = ("You fill in missing values in rows of a data table. Reply with exactly one line per missing "
                "column, in the form <column>: <value>, and nothing else. If a value cannot be inferred, "
                "write unknown as its value.")

PROMPT_VERSION = hashlib.sha256("\x1f".join([SYSTEM_INSTRUCT, SYSTEM_BATCH, "v2",
                                             f"P1:{RAW_ROLE}:{RAW_STOP!r}"]).encode()).hexdigest()[:12]


ABSTAIN = {"unknown", "n/a", "na", "none", "null", "?", "unk", "cannot determine", "not available",
           "cannot be inferred", "can't determine", "not enough information", "insufficient information"}


@dataclass
class PromptInputs:
    row: int
    columns: list[str]
    context_rows: list[int]


def _serialize(items: list[tuple[str, str]]) -> str:
    return ". ".join(f"{c}: {v}" for c, v in items) + ("." if items else "")


def _type_word(mt: MaskedTable, c: str) -> str:
    return "a number" if mt.numeric[c] else "a text value"


def _candidate_lines(mt: MaskedTable, columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        if mt.numeric[c]:
            s = mt.numeric_stats.get(c)
            if s:
                out.append(f"Observed values of {c}: min {format_value(s['min'], True)}, "
                           f"median {format_value(s['median'], True)}, max {format_value(s['max'], True)}")
        else:
            dom = mt.domains.get(c, [])[:MAX_CANDIDATES]
            if dom:
                out.append(f"Candidate values for {c} (most frequent observed values): " + " | ".join(dom))
    return out


def build_messages(mt: MaskedTable, inp: PromptInputs, prompt: str) -> list[dict]:
    if prompt not in ("P1", "P2", "P3", "P4"):
        raise ValueError(prompt)
    if prompt != "P4" and len(inp.columns) != 1:
        raise ValueError("P1-P3 impute exactly one cell per request")
    if len(inp.columns) > P4_MAX_CELLS:
        raise ValueError("P4 batches hold at most 5 cells")
    targets = set(inp.columns)
    row_items = mt.row_items(inp.row, None, exclude=targets)

    if prompt == "P1":
        (c,) = inp.columns
        lines = []
        for r in inp.context_rows:
            ans = mt.display.at[r, c]
            ex = _serialize(mt.row_items(r, None, exclude=targets))
            lines.append(f"{ex} What is the {c}? {ans}")
        if lines:
            lines.append("")
        lines.append(f"{_serialize(row_items)} What is the {c}?")
        return [{"role": RAW_ROLE, "content": "\n".join(lines)}]

    batched = prompt == "P4"
    system = SYSTEM_BATCH if batched else SYSTEM_INSTRUCT
    body: list[str] = []
    if inp.context_rows:
        body.append("Example rows from the same table:")
        for r in inp.context_rows:
            body.append("- " + _serialize(mt.row_items(r, None)))
        body.append("")
    body.append("Row with missing values:" if batched else "Row with a missing value:")
    body.append(_serialize(row_items))
    if prompt in ("P3", "P4"):
        body += _candidate_lines(mt, inp.columns)
    if batched:
        body.append("Missing columns: " + ", ".join(inp.columns))
    else:
        (c,) = inp.columns
        body.append(f"Missing column: {c} ({_type_word(mt, c)})")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(body)}]


@dataclass
class Parsed:
    value: str | float | None = None
    raw: str | None = None
    abstained: bool = False
    error: str | None = None


_NUMBER = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_PREFIX = re.compile(r"^(?:answer|value|the answer is|final answer)\s*[:=-]\s*", re.I)


def _clean_text(s: str) -> str:
    s = s.strip().strip("`").strip()
    s = _PREFIX.sub("", s)
    for _ in range(2):
        if s.endswith(".") and not _NUMBER.fullmatch(s):
            s = s[:-1].rstrip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            s = s[1:-1].strip()
    return s


def type_value(text: str, numeric: bool) -> Parsed:
    t = _clean_text(text)
    if t.lower() in ABSTAIN:
        return Parsed(raw=t, abstained=True)
    if numeric:
        m = _NUMBER.search(t)
        if not m:
            return Parsed(raw=t, error="C2: not a number")
        return Parsed(value=float(m.group(0).replace(",", "")), raw=t)
    if not t:
        return Parsed(raw=t, error="C2: empty value")
    if len(t) > 200:
        return Parsed(raw=t[:200], error="C2: value longer than 200 characters")
    return Parsed(value=t, raw=t)


def parse_single(text: str, numeric: bool, column: str) -> Parsed:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return Parsed(raw="", error="C2: empty answer")
    s = lines[0].strip()
    pref = re.match(rf"^\s*{re.escape(column)}\s*:\s*", s, re.I)
    if pref:
        s = s[pref.end():]
    return type_value(s, numeric)


def parse_batch(text: str, columns: list[str], numeric: dict[str, bool]) -> dict[str, Parsed] | None:
    found: dict[str, list[str]] = {c: [] for c in columns}
    lookup = {c.lower(): c for c in columns}
    for ln in (text or "").splitlines():
        m = re.match(r"^\s*[-*]?\s*([^:]+?)\s*:\s*(.*)$", ln)
        if not m:
            continue
        key = m.group(1).strip().strip("*`").lower()
        if key in lookup:
            found[lookup[key]].append(m.group(2))
    if len(columns) == 1 and not found[columns[0]]:
        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        if len(lines) == 1:
            found[columns[0]] = [lines[0]]
    if any(len(v) != 1 for v in found.values()):
        return None
    return {c: type_value(found[c][0], numeric[c]) for c in columns}
