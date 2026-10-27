# DFT candidate selection criteria

Written on 2026-10-28, before the filter was run and before any DFT calculation. If a rule changes later, the
change goes in its own commit with the reason, so the history shows what was decided before the results were
seen.

## Input

- `results/v3/screened_candidates.csv`: compositions that the v3 model (`models/v3`, net-moment target,
  isotonic-calibrated) scores at or above its operating threshold. None of them is in `data/v3/known_materials.txt`
  (all V2DB and thesis C2DB formulas).
- For comparison only: the thesis top 10 (`results/legacy/novel_candidates.csv`), run through the same rules.

## Filters, in order

Each candidate gets a pass/fail for every filter, and the first filter it fails is recorded. The local filters
(S1-S4, S8) are evaluated for every candidate. The network filters (S5 and S6) are evaluated for the candidates
that pass S1-S4, and for the thesis top 10.

| # | Filter | Rule |
| --- | --- | --- |
| S1 | Score | Calibrated v3 score ≥ 0.98. On the C2DB test rows, this band had a measured precision of 187/193. |
| S2 | In distribution | The candidate's anonymous formula (e.g. `A2BC4` for Mn2CrS4) occurs at least 20 times among the training compositions. Below that, the score comes from a region the model has hardly seen. |
| S3 | Charge balance | Some assignment of **one** oxidation state per element gives zero net charge. States come from pymatgen `Element.icsd_oxidation_states`, or `common_oxidation_states` if the ICSD list is empty. |
| S4 | Electronegativity | In that same assignment, every element with a positive state has a lower Pauling electronegativity than every element with a negative state, and there is at least one negative state. S3 + S4 follow the SMACT screening rule (Davies et al., *Chem* 1, 617, 2016). |
| S5 | Not in a 2D database | Exact `chemical_formula_reduced` match in MC2D or 2DMatPedia (OPTIMADE), or in the thesis C2DB table, means the 2D compound is already known: fail. A match in Materials Project, Alexandria (PBE) or JARVIS-DFT is recorded as `bulk_known` but does not fail. |
| S6 | Not in the 2D literature | OpenAlex (title and abstract) and arXiv (all fields) are searched for the formula in each ordering of its cations. If any returned work has the formula together with a 2D word (monolayer, two-dimensional, 2D, single-layer, nanosheet, few-layer, exfoliat*) in its title or abstract, it fails. Hits without a 2D word are recorded as `literature_bulk` and pass. |
| S7 | Family diversity | Family = anonymous formula + set of anions. At most 2 candidates per family among the selected 10, taking the highest scores first. |
| S8 | A 2D structure can be built | At least one 2D entry with the same anonymous formula exists in MC2D or 2DMatPedia, to serve as a prototype for element substitution (stage P). |

Ranking: calibrated score, then the raw boosting score. That order breaks ties, because isotonic calibration
flattens the top of the scale.

## Groups

- **Main (10):** the 10 highest-ranked candidates that pass S1-S8. These go to DFT: ISIF=3 relax, ISIF=2 relax,
  then phonons.
- **Reference:** CrI3 as the protocol benchmark. It has a known FM ground state and stable phonons in the
  literature. Up to 2 more come from the highest-scoring candidates that failed only S5 or S6: their 2D form is
  known, so our numbers can be compared with published ones.
- **Thesis top 10:** listed with the filter where each one fails. Not calculated.

A candidate that turns out unstable in DFT stays in the results. "k of the 10 are dynamically stable" is a
finding.

## Known limitations

- One oxidation state per element: mixed-valence compounds (Fe3O4-like) fail S3.
- The literature search can miss papers that write the formula with another element order, with subscripts that
  break the token, or only in the full text. Before the main 10 are fixed, I check each one by hand in Web of
  Science and Google Scholar, and record the result.
- On 2026-10-28, the CMR (C2DB) OPTIMADE endpoint redirected away and OQMD returned 502. C2DB is covered through
  the thesis C2DB table, OQMD is not used.
- Database responses are cached under `dft/cache/` with the query date. A later rerun can differ if the databases
  change.
