# Processed data

Aggregated tables built by `scripts/`. There's no individual-level data here: every row is a
total or an average over many survey respondents, the same kind of number government agencies
publish. The IPUMS CPS microdata these come from can't be redistributed, so to rebuild them you
download your own extract (see `docs/IPUMS_EXTRACT.md`).

| File | One row per | Built by |
|---|---|---|
| `panel_occ_age_quarter.parquet` | occupation group × age group × quarter (CPS) | `build_panel.py` |
| `panel_occ_sector_year.parquet` | occupation group × industry sector × age band × year (CPS) | `build_occ_sector_panel.py` |
| `ces_panel.parquet`, `ces_industries.csv` | industry × month (BLS payroll data), and each industry's traits | `build_industry_panel.py` |
| `occ_groups.csv` | occupation group, with its AI exposure scores | `build_panel.py` |
| `exposure_occ2010.csv` | detailed occupation code, with its AI exposure scores | `build_exposure_crosswalk.py` |
| `occ_rate_sensitivity*.csv` | occupation group, with its estimated rate sensitivity and cyclicality | `estimate_rate_sensitivity.py` |

In this public copy, average earnings are blanked for the few cells with fewer than 3
respondents, so no cell shows one person's earnings. No reported result uses that column.
