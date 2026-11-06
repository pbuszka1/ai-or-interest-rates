# Analysis Notebooks

One short notebook per analysis. Each one gives the question, how it works, the key numbers,
the chart and what it means. They open with results already shown; Run All takes seconds.
To recompute from the data, set `RUN = True` at the top (needs the raw data, which isn't in
this repo; see the README).

| # | Notebook | One-line result |
|---|---|---|
| 1 | [Synthetic control: computer occupations](01_synthetic_control_occupations.ipynb) | Failed its hold-out check; a documented negative result |
| 2 | [Synthetic control: computer systems design industry](02_synthetic_control_industry.ipynb) | About −13% by 2026, but p = 0.29: inconclusive |
| 3 | [Synthetic DiD: high vs low exposure industries](03_synthetic_did_industries.ipynb) | No AI-specific decline against a 2019 baseline |
| 4 | [Hires by age (QWI)](04_hires_by_age_qwi.ipynb) | Young hiring squeezed about 9%, but in 2020–21, before ChatGPT |
| 5 | [Young vs older workers in exposed occupations](05_young_vs_old_occupations.ipynb) | Borderline; mostly pre-ChatGPT, possible further drop in 2026 |
| 6 | [Injection tests](06_injection_tests.ipynb) | Industry design detects about 5% effects; age designs need about 20–24% |
| 7 | [Test A: dose-response across industries](07_test_A_dose_response_industries.ipynb) | Pre-registered; failed its pre-trend check, so no verdict |
| 8 | [Test B: occupations inside the same industry](08_test_B_within_industry.ipynb) | Exposed jobs haven't lost ground within industries (±6%); young-worker signal to watch |

Open with `jupyter lab notebooks` after `pip install -r requirements.txt`, or in VS Code.
The whole story in prose: [../README.md](../README.md).
