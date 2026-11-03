# How to recreate the IPUMS CPS extract

The IPUMS license doesn't allow redistributing the microdata, so this repo contains the
recipe instead. Following it gives you the same file the analysis uses (extract `cps_00001`,
created 2026-10-21).

1. Register (free) at [cps.ipums.org](https://cps.ipums.org).
2. **Samples:** Basic Monthly only, every month from **January 1990 to September 2026**
   (440 samples; October 2025 was never collected because of the federal government
   shutdown). No ASEC samples.
3. **Variables** (harmonized versions):

   | Group | Variables |
   |---|---|
   | Demographics | AGE, SEX, RACE, EDUC, NATIVITY, CITIZEN, SCHLCOLL |
   | Work status | EMPSTAT, LABFORCE, CLASSWKR, WKSTAT, UHRSWORKT |
   | Occupation and industry | OCC, OCC2010, OCC1990, IND, IND1990 |
   | Job changes | EMPSAME, ACTSAME |
   | Unemployment | WHYUNEMP, DURUNEMP, DURUNEM2 |
   | Earnings | EARNWEEK, HOURWAGE2, EARNWT |
   | Survey design | MISH, STATEFIP, LNKFW1MWT |
   | Optional (October supplement) | EDFULL, EDTYPE |

   IPUMS adds the identifiers and weights automatically (YEAR, MONTH, SERIAL, PERNUM, CPSID,
   CPSIDP, CPSIDV, HWTFINL, WTFINL, ASECFLAG) and the data quality flags (Q-prefixed).
4. **Case selection:** AGE 18 to 65 only. The extract keeps everyone in that age range, in or out of the labor force (and armed forces members, LABFORCE = 0); the code selects the employed using EMPSTAT.
5. **Format:** fixed-width `.dat`, rectangular (person) structure. No CPI adjustment:
   earnings are converted to constant dollars in code (`scripts/`), so the method is visible.
6. Download the `.dat.gz` file and the **DDI** codebook (`.xml`) into `data/raw/`, then run
   `scripts/load_cps.py` followed by `scripts/validate_cps.py`.

The extract number will differ from `00001`; pass yours to the loader:
`python scripts/load_cps.py 00007`.
