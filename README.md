# LLM Data Wrangler: pre-registration

Pre-registration of the study "Evolving Lightweight LLM Pipelines for Cost-Aware Missing-Value Imputation in Tabular Data" (CITF7001, Istanbul University, Fall 2026). The tag `prereg-2026-11-18` marks the registered version.

## What is registered and where

| Item | File |
|---|---|
| Prompt templates P1 to P4 | `src/llm_wrangler/prompts.py` |
| Genome and founders | `configs/genome.yaml` |
| Population, children per generation, generations | `configs/experiment.yaml`, section `evolution` |
| Fitness weights | `configs/experiment.yaml`, section `fitness` |
| Token and time budget per table | `configs/budgets.yaml`, rule in `configs/experiment.yaml`, section `budget` |
| Model and backup model | `configs/model.yaml`, `configs/local-models.yaml` |
| Missingness injection | `src/llm_wrangler/inject.py`, `scripts/make_masks.py` |
| Masks and seeds | `masks/` with `manifest.json` (SHA-256 of every mask); seeds in `configs/experiment.yaml`, section `missingness` |

## Regenerating the masks

The tables themselves are not copied here. Put each table's `clean.csv` under `data/raw/<table>/` (Hospital and Beers from the Raha repository, Restaurant converted from the R package `restaurant`) and run

    python scripts/make_masks.py --tables hospital beers restaurant

with Python 3.12, numpy, pandas, pyarrow and pyyaml. The SHA-256 values in `masks/manifest.json` show whether the result matches the registered masks.
