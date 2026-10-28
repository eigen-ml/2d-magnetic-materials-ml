"""Stage P, second half: rank the starting structures with MACE-MP-0.

Each POSCAR from 07_build_structures.py is relaxed with MACE (atoms and in-plane cell; the
vacuum axis is held). Per candidate, prototypes are compared by energy per atom. The lowest is
marked for VASP, and the second too when it lies within --window. MACE only chooses which
prototype goes to VASP; it ignores spin order, and no MACE number is reported as a result.

Writes POSCAR.mace next to each POSCAR, adds the MACE fields to meta.json, and writes
dft/structures/summary_mlip.csv. Runs on CPU in the separate .venv-mlip environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

ROOT = Path(__file__).resolve().parents[1]


def thickness(atoms) -> float:
    """Slab thickness along the normal of the a-b plane."""
    a, b = atoms.cell[0], atoms.cell[1]
    normal = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    heights = atoms.positions @ normal
    return float(heights.max() - heights.min())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--structures", type=Path, default=ROOT / "dft" / "structures")
    parser.add_argument("--fmax", type=float, default=0.05, help="force threshold, eV/A")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--window", type=float, default=0.025, help="second prototype also goes if within this, eV/atom")
    args = parser.parse_args()

    calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
    rows = []
    for poscar in sorted(args.structures.glob("*/*/POSCAR")):
        atoms = read(poscar, format="vasp")
        start_thickness = thickness(atoms)
        atoms.calc = calc
        # strain mask in Voigt order xx yy zz yz xz xy: only in-plane components move
        relaxer = BFGS(FrechetCellFilter(atoms, mask=[1, 1, 0, 0, 0, 1]), logfile=None)
        converged = bool(relaxer.run(fmax=args.fmax, steps=args.steps))
        energy = float(atoms.get_potential_energy()) / len(atoms)
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        write(poscar.with_name("POSCAR.mace"), atoms, format="vasp", direct=True, sort=True)
        result = {
            "mace_model": "MACE-MP-0 medium",
            "mace_energy_eV_per_atom": round(energy, 5),
            "mace_converged": converged,
            "mace_steps": relaxer.nsteps,
            "mace_min_distance_A": round(float(distances.min()), 3),
            "mace_thickness_A": round(thickness(atoms), 3),
            "start_thickness_A": round(start_thickness, 3),
            "mace_a_b_A": [round(float(x), 3) for x in atoms.cell.lengths()[:2]],
        }
        meta_path = poscar.with_name("meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(result)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"formula": meta["formula"], "group": meta["group"], "folder": poscar.parent.name,
                     "spacegroup_start": meta["spacegroup_symprec0.1"], **result})
        print(f"{meta['formula']:10s} {poscar.parent.name[:40]:40s} {energy:9.4f} eV/atom  conv={converged}")

    table = pd.DataFrame(rows)
    table["delta_meV_per_atom"] = 1000 * (
        table["mace_energy_eV_per_atom"] - table.groupby("formula")["mace_energy_eV_per_atom"].transform("min")
    )
    table = table.sort_values(["formula", "delta_meV_per_atom"])
    table["mace_rank"] = table.groupby("formula").cumcount() + 1
    table["to_vasp"] = (table["mace_rank"] == 1) | (
        (table["mace_rank"] == 2) & (table["delta_meV_per_atom"] <= 1000 * args.window)
    )
    table.drop(columns="mace_a_b_A").to_csv(args.structures / "summary_mlip.csv", index=False)


if __name__ == "__main__":
    main()
