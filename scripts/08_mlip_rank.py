"""Stage P, second half: rank the starting structures with MACE-MP-0.

Each POSCAR from 07_build_structures.py is relaxed with MACE (atoms and in-plane cell; the
vacuum axis is held). Per candidate, prototypes are compared by energy per atom. The lowest is
marked for VASP, and the second too when it lies within --window, provided the relaxed layer is
still one bonded piece without a thickness jump (layer_holds). MACE only chooses which
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
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import BFGS
from mace.calculators import mace_mp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]


def thickness(atoms) -> float:
    """Slab thickness along the normal of the a-b plane."""
    a, b = atoms.cell[0], atoms.cell[1]
    normal = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    heights = atoms.positions @ normal
    return float(heights.max() - heights.min())


def layer_check(atoms, bond_scale: float = 1.2) -> tuple[int, float]:
    """Bonded pieces in the cell (covalent radii x bond_scale) and the largest gap between atomic planes, in A.

    One piece means the monolayer held together; more than one means it split into sublayers or molecules.
    """
    i, j = neighbor_list("ij", atoms, [bond_scale * r for r in natural_cutoffs(atoms)])
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms), len(atoms)))
    pieces = connected_components(graph, directed=False)[0]
    a, b = atoms.cell[0], atoms.cell[1]
    normal = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    heights = np.sort(atoms.positions @ normal)
    gap = float(np.diff(heights).max()) if len(heights) > 1 else 0.0
    return int(pieces), gap


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
        pieces, gap = layer_check(atoms)
        result = {
            "mace_model": "MACE-MP-0 medium",
            "mace_energy_eV_per_atom": round(energy, 5),
            "mace_converged": converged,
            "mace_steps": relaxer.nsteps,
            "mace_min_distance_A": round(float(distances.min()), 3),
            "mace_thickness_A": round(thickness(atoms), 3),
            "start_thickness_A": round(start_thickness, 3),
            "mace_a_b_A": [round(float(x), 3) for x in atoms.cell.lengths()[:2]],
            "mace_bonded_pieces": pieces,
            "mace_max_plane_gap_A": round(gap, 3),
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
    table["thickness_ratio"] = (table["mace_thickness_A"] / table["start_thickness_A"]).round(3)
    # the layer holds: one bonded piece, no thickness jump; a split prototype is not sent to VASP
    table["layer_holds"] = (table["mace_bonded_pieces"] == 1) & table["thickness_ratio"].between(0.7, 1.3)
    table = table.sort_values(["formula", "delta_meV_per_atom"])
    table["mace_rank"] = table.groupby("formula").cumcount() + 1
    # VASP gets the lowest prototype that holds, and the next one that holds if it is within the window
    held = table[table["layer_holds"]]
    held_rank = held.groupby("formula").cumcount() + 1
    held_delta = 1000 * (held["mace_energy_eV_per_atom"]
                         - held.groupby("formula")["mace_energy_eV_per_atom"].transform("min"))
    table["to_vasp"] = False
    table.loc[held.index, "to_vasp"] = (held_rank == 1) | ((held_rank == 2) & (held_delta <= 1000 * args.window))
    table.drop(columns="mace_a_b_A").to_csv(args.structures / "summary_mlip.csv", index=False)


if __name__ == "__main__":
    main()
