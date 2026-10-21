# legacy/ — archived SAE phase-segmentation direction

Deprecated on 2026-10-21 when the project pivoted to *Self-Sourced Repair*
(see `docs/DECISIONS.md`). These modules (`sae`, `segmentation`, `taxonomy`,
`metrics`) are unchanged apart from their import paths
(`rfd.X` -> `rfd.legacy.X`); their tests live in `tests/legacy/` and carry the
`legacy` pytest marker. Reproduce the old pilot with `make legacy-pilot`
(outputs in `results/legacy_pilot/`).
