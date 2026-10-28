# DFT validation: candidate selection

The rules are in [`CRITERIA.md`](CRITERIA.md). They were written before the filter was run, and the changes
made after the first run (before any DFT) are listed there with their reasons. The filter is
`scripts/06_filter_candidates.py`.

## Funnel

| Step | Left |
| --- | --- |
| v3 screening (score above the model threshold) | 1,577 |
| S1 calibrated score ≥ 0.98 | 759 |
| S2 stoichiometry well represented in training | 759 |
| S3 charge balance (ICSD ≥ 1% states) | 372 |
| S4 electronegativity order | 364 |
| S5 not in a 2D database | 361 |
| S6 not in the 2D literature | 350 |
| S8 a 2D prototype exists | 350 |
| S9 bulk hull distance ≤ 0.1 eV/atom (Alexandria) | 208 |
| S7 at most 2 per family and per cation set, top 10 | **10** |

## Main 10

| Candidate | Oxidation states | Bulk hull, Alexandria (eV/atom) | Family |
| --- | --- | --- | --- |
| Mn2VSe4 | Mn+2 V+4 Se-2 | 0.000 | A4B2C, Se |
| Mn2VS4 | Mn+2 V+4 S-2 | 0.000 | A4B2C, S |
| Mn2CoSe4 | Mn+3 Co+2 Se-2 | 0.025 | A4B2C, Se |
| Mn2CoTe4 | Mn+3 Co+2 Te-2 | 0.036 | A4B2C, Te |
| Mn2CrS4 | Mn+3 Cr+2 S-2 | 0.028 | A4B2C, S |
| Mn2CrTe4 | Mn+3 Cr+2 Te-2 | 0.014 | A4B2C, Te |
| MnFeS3 | Mn+3 Fe+3 S-2 | 0.075 | A3BC, S |
| MnBr2Cl | Mn+3 Br-1 Cl-1 | 0.033 | A2BC, Br+Cl |
| MnBrCl2 | Mn+3 Br-1 Cl-1 | 0.019 | A2BC, Br+Cl |
| CrCoSe3 | Cr+3 Co+3 Se-2 | 0.066 | A3BC, Se |

All 10 have a calibrated score of at least 0.989. None of them matched in MC2D, 2DMatPedia, Materials Project or
JARVIS, or in the literature search. They do appear in Alexandria, a database of computed (mostly hypothetical)
bulk structures, which is where the hull distances come from.

**References:** CrI3 (protocol benchmark) and VCl3. Both are known 2D magnets with published DFT values.

**Thesis top 10:**
- 8 fail charge balance.
- Mn2FeO4 fails S6: the 2D literature exists.
- Mn2CrO4 passes every filter but does not make the 10. Its cation set {Cr, Mn} is already taken by Mn2CrS4 and
  Mn2CrTe4, which score higher.

Every candidate, with each filter result and its evidence, is in `candidates_final.csv`. The funnel and groups
are in `selection.json`.

## Before DFT

1. I check each of the 10 by hand in Web of Science and Google Scholar. The automatic search tried one anion
   order only, so MnBr2Cl and MnBrCl2 especially need this.
2. Stage P: build structures from same-stoichiometry 2D prototypes, relax them with an ML potential to pick the
   prototype, and write POSCARs with 20 Å vacuum. Ata and Yeşim Hoca look at them in VESTA before anything goes
   to TRUBA.

## Caveats

- Mn is in 9 of the 10, and A2BX4 in 6. This is the model's preference, after the filters.
- Element-level ICSD counts ignore the anion: Mn+3 is common in oxides, less so in chalcogenides and halides.
- A composition near the bulk hull says nothing yet about its monolayer. The relax and phonon calculations
  answer that.
- Database and literature answers are cached in `dft/cache/` (not tracked). A rerun on another date can differ.
