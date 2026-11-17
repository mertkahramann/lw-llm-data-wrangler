#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import yaml
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_wrangler.data import load_table
from llm_wrangler.inject import (MECHANISMS, make_mask, mask_name, mask_report,
                                 select_targets)

RATES = (0.10, 0.20, 0.30)
TEST_SEEDS = (0, 1, 2, 3, 4)
DEV = (("MCAR", 0.20, 100), ("MAR", 0.20, 101), ("MNAR", 0.20, 102))
PILOT = {"hospital": ("MCAR", 0.10, 999)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()

    manifest_path = ROOT / "masks" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"masks": {}}
    manifest["spec"] = "v4.3.1 §4.3, §4.4a"
    manifest["generator"] = "scripts/make_masks.py"
    exp = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    mnar_rule = exp["missingness"].get("mnar_rule", "frozen")
    if mnar_rule != "frozen":
        manifest["mnar_rule"] = mnar_rule

    for t in args.tables:
        table = load_table(t, Path(args.raw_dir) / t / "clean.csv")
        out_dir = ROOT / "masks" / t
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(m, r, s, "test") for m in MECHANISMS for r in RATES for s in TEST_SEEDS]
        jobs += [(m, r, s, "development") for m, r, s in DEV]
        if t in PILOT:
            m, r, s = PILOT[t]
            jobs.append((m, r, s, "pilot"))
        for mech, rate, seed, role in jobs:
            mask = make_mask(table, mech, rate, seed, mnar_rule=mnar_rule)
            sel = select_targets(table, mask, mech, rate, seed)
            name = mask_name(mech, rate, seed)
            mpath, tpath = out_dir / f"{name}.parquet", out_dir / f"{name}_targets.parquet"
            mask.to_parquet(mpath, index=False)
            sel.targets.to_parquet(tpath, index=False)
            entry = mask_report(table, mask, sel, mech, rate, seed)
            entry.update({"role": role, "mask_file": mpath.relative_to(ROOT).as_posix(),
                          "targets_file": tpath.relative_to(ROOT).as_posix(),
                          "mask_file_sha256": file_sha256(mpath),
                          "targets_file_sha256": file_sha256(tpath)})
            manifest["masks"][f"{t}/{name}"] = entry
        print(f"{t}: {len(jobs)} masks, eligible cells {len(table.eligible_cells())}, "
              f"targets {table.targets}")

    manifest["masks"] = dict(sorted(manifest["masks"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path.relative_to(ROOT)} ({len(manifest['masks'])} masks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
