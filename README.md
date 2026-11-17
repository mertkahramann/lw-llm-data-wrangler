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

## Data

The three tables are in `data/raw/<table>/clean.csv`. Hospital and Beers come from the Raha repository (Apache-2.0). The Beers data is J.-N. Hould's Craft Beers Dataset (DbCL v1.0 on Kaggle, MIT on GitHub). Hospital goes back to the CMS Hospital Compare data, which is public-use with no licence stated. Restaurant is converted from the R package `restaurant` (CC0).

## Regenerating the masks

    python scripts/make_masks.py --tables hospital beers restaurant

with Python 3.12, numpy, pandas, pyarrow and pyyaml. The SHA-256 values in `masks/manifest.json` show whether the result matches the registered masks.
