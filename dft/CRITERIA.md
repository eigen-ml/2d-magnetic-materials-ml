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
| S3 | Charge balance | Some assignment of **one** oxidation state per element gives zero net charge. Allowed states are those that make up at least 1% of the element's non-zero oxidation-state entries in ICSD (2024), from the counts shipped with SMACT 4.0.2 (`oxidation_states_icsd24_counts.json`). *(Amended 2026-10-28, see below.)* |
| S4 | Electronegativity | In that same assignment, every element with a positive state has a lower Pauling electronegativity than every element with a negative state, and there is at least one negative state. S3 + S4 follow the SMACT screening rule (Davies et al., *Chem* 1, 617, 2016). |
| S5 | Not in a 2D database | Exact `chemical_formula_reduced` match in MC2D or 2DMatPedia (OPTIMADE), or in the thesis C2DB table, means the 2D compound is already known: fail. A match in Materials Project, Alexandria (PBE) or JARVIS-DFT is recorded as `bulk_known` but does not fail. |
| S6 | Not in the 2D literature | OpenAlex (title and abstract) and arXiv (all fields) are searched for the formula in each ordering of its cations. If any returned work has the formula together with a 2D word (monolayer, two-dimensional, 2D, single-layer, nanosheet, few-layer, exfoliat*) in its title or abstract, it fails. Hits without a 2D word are recorded as `literature_bulk` and pass. |
| S7 | Diversity | Family = anonymous formula + set of anions. Among the selected 10, at most 2 per family **and** at most 2 per cation set (e.g. {Mn, V}), highest scores first. *(Cation cap added 2026-10-28.)* |
| S8 | A 2D structure can be built | At least one 2D entry with the same anonymous formula exists in MC2D or 2DMatPedia, to serve as a prototype for element substitution (stage P). |
| S9 | Bulk stability | The lowest Alexandria (PBE) hull distance of the same composition in bulk is at most 0.1 eV/atom. With no Alexandria entry the candidate passes, and the missing value is visible in the table. A 2D sheet can sit a little above its bulk, but a composition far above the hull in every bulk structure is unlikely to be stable as a monolayer. *(Added 2026-10-28.)* |

Ranking: calibrated score, then the raw boosting score. That order breaks ties, because isotonic calibration
flattens the top of the scale.

## Groups

- **Main (10):** the 10 highest-ranked candidates that pass S1-S9. These go to DFT: ISIF=3 relax, ISIF=2 relax,
  then phonons.
- **Reference:** CrI3 and VCl3. Both are known 2D magnets with published DFT numbers. CrI3 checks the protocol;
  VCl3 is close to the ABX3 candidates. *(Fixed 2026-10-28; this replaces automatic picks.)*
- **Thesis top 10:** listed with the filter where each one fails. Not calculated.

A candidate that turns out unstable in DFT stays in the results. "k of the 10 are dynamically stable" is a
finding.

## Known limitations

- One oxidation state per element: mixed-valence compounds (Fe3O4-like) fail S3.
- The literature search can miss papers that write the formula with another element order, with subscripts that
  break the token, or only in the full text. Before the main 10 are fixed, I check each one by hand in Web of
  Science and Google Scholar, and record the result.
- On 2026-10-28, the CMR (C2DB) OPTIMADE endpoint redirected away and OQMD returned 502. C2DB is covered through
  the thesis C2DB table, OQMD is not used. *(2026-10-30)* The thesis table has the same 13,251 formulas as the
  current C2DB file (`c2db.db`, 2025-07-08, see `results/c2db_audit/summary.json`). S5 run against that file gives
  a byte-identical `candidates_final.csv`. The C2DB web search listed 17,001 rows on 2026-10-30, against 16,905
  structures in the file. The extra 96 were not checked.
- The C2DB label means "spin-polarized in the FM state", not "ferromagnetic ground state": C2DB stores
  antiferromagnets such as MnPS3 and MnBr2 in their FM state. The score says nothing about FM versus AFM. The DFT
  step that compares NM, FM and AFM orderings answers that.
- Database responses are cached under `dft/cache/` with the query date. A later rerun can differ if the databases
  change.

## Amendments

**2026-10-28, after the first filter run and before any DFT calculation.** The first run is commit `e2aa320`,
kept unchanged. Ata decided these three changes after reviewing it:

1. **S3 oxidation states.** Before, every state in pymatgen's ICSD list was allowed. Four of the first main 10
   (MnCoBr3, MnCo2Br4, MnCoCl3, MnCo2Cl4) balanced only with Co+1, which is 0.8% of cobalt's ICSD entries. Now a
   state must make up at least 1% of the element's entries. The 1% cut was chosen as the loosest one that removes
   the rare tail. The top of the ranking is the same at 1%, 2% and 5%. A limit remains: element-level counts do
   not see the anion, so Mn+4 (13.5% of Mn entries, mostly oxides) is still allowed in selenides and tellurides.
2. **S7 cation cap.** All of the first main 10 contained Mn, and three were Mn-V. Adding at most 2 per cation set
   spreads the 10 calculations over more metal pairs. The model's preference for Mn stays visible in the full
   table and is reported.
3. **Reference group.** The automatic extra references (Mn2CoO4, Mn2CoS4) had failed S6 on electrochemistry
   papers about "nanosheets", which give no 2D magnetic numbers to compare with. The references are now CrI3
   and VCl3.
4. **S9 bulk stability.** After the rerun (`ba182d8`), four of the main 10 sat far above the bulk hull in
   Alexandria: MnFeTe3 0.78, MnFeSe3 1.37, MnNiBr3 0.68 and MnNiCl3 0.67 eV/atom. MnFeTe3 and MnFeSe3 also have
   positive formation energies. Spending relax and phonon time on them is unlikely to pay off, so S9 drops
   compositions above 0.1 eV/atom. 154 of the 350 candidates that passed S1-S8 are at or below it. This takes
   the place of the planned MLIP pre-screen as a filter; MLIP relaxation moves to stage P, where it picks the
   prototype.
5. **DFT batch 1 goes ahead before the Mn diagnosis.** Nine of the main 10 contain Mn. A first check suggests the
   model takes this from V2DB, whose ML labels call Mn compounds magnetic far more often than Cr, Fe or Co ones
   (C2DB does not show the gap). The diagnosis is not finished. Batch 1 is the main 10 and the two references
   exactly as listed in `dft/selection.json` at commit `fdaac55`. It is not edited or trimmed after the
   diagnosis. If the diagnosis changes the model, the new list is batch 2 and is reported next to batch 1. Every
   batch 1 result is reported, failures included. Reason: relax and phonon judge each structure on its own, so a
   biased ranking changes which candidates get computed, not whether a computed one is stable. Cost: batch 1
   says little about how well the model ranks non-Mn compounds.
   *Correction, 2026-10-30:* "C2DB does not show the gap" was wrong. On the C2DB test rows, 95% of the Mn
   compositions are magnetic (Cr 86%, Fe 86%, V 64%, Co 56%, Ni 41%), so the preference comes from the C2DB
   label itself. V2DB mainly pushes Cr down. Batch 1 stays as fixed above. Diagnosis: `results/v3/README.md`.

**2026-10-30, before any DFT calculation.** Ata decided these after the Mn diagnosis and the C2DB label check:

6. **Control group next to batch 1.** 3-4 compositions with a v4 score below 0.3 that pass S3 and S8, plus
   MnBr2. MnBr2 is an antiferromagnetic monolayer in the literature, but C2DB stores it in the FM state with
   5.0 μB. The group gets relax and the NM/FM/AFM comparison, no phonons. It measures two things: whether the
   model separates low from high scores, and whether the DFT protocol finds an AFM ground state where there is
   one. The list is fixed in `dft/selection.json` before any calculation of the group, and every result is
   reported.
7. **Main model and batch 2.** v4, trained on the C2DB rows only (DFT labels), is the main model. On the C2DB
   test rows it reaches ROC-AUC 0.939 against 0.893 for v3. Batch 2 comes from v4 with the same filters and at
   most 3 compositions per magnetic metal. v3 and the metal rule are reported next to it.
8. **S5 reference.** The thesis table and the C2DB file from 2025-07-08 have the same formula set, so S5 keeps
   the thesis table as its default and batch 1 does not change. A newer C2DB release goes through
   `scripts/09_c2db_audit.py` first.
