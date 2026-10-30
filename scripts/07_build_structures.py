"""Stage P: starting structures for a DFT batch and its control group.

For each candidate, 2D structures with the same anonymous formula are taken from MC2D and
2DMatPedia (OPTIMADE). Elements are put on the prototype sites by count, sign of the
oxidation state and electronegativity order. The slab is scaled by the ratio of ionic radii,
centred, and given 20 A of vacuum along a c axis normal to the layer. Prototypes are ranked
by how far the oxidation states and radii have to move; the best few are written.

Output: dft/structures/<formula>/<rank>_<source>_<id>/POSCAR and meta.json, plus
dft/structures/summary.csv. These are starting points for MLIP pre-relaxation and a VESTA
check, not relaxed structures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Element, Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("filters", ROOT / "scripts" / "06_filter_candidates.py")
filters = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filters)

SOURCES = {
    "mc2d": (
        "https://optimade.materialscloud.org/main/mc2d",
        "Campi et al., ACS Nano 17, 11268 (2023); Mounet et al., Nat. Nanotechnol. 13, 246 (2018)",
        "CC BY 4.0",
    ),
    "twodmatpedia": (
        "http://optimade.2dmatpedia.org",
        "Zhou et al., Sci. Data 6, 86 (2019)",
        "CC BY 4.0",
    ),
}
FIELDS = "id,chemical_formula_reduced,lattice_vectors,cartesian_site_positions,species_at_sites,dimension_types"
VACUUM = 20.0


def fetch_prototypes(client: Any, base_url: str, anonymous: str, max_pages: int = 20) -> list[dict]:
    """Every structure with this anonymous formula, following pagination."""
    params = {"filter": f'chemical_formula_anonymous="{anonymous}"', "response_fields": FIELDS, "page_limit": 100}
    response = client.get(base_url.rstrip("/") + "/v1/structures", params)
    items: list[dict] = []
    for _ in range(max_pages):
        if not isinstance(response, dict) or "data" not in response:
            break
        items += response["data"]
        next_link = response.get("links", {}).get("next")
        if isinstance(next_link, dict):
            next_link = next_link.get("href")
        if not next_link:
            break
        response = client.get(next_link, {})
    return items


def radius(symbol: str, state: int) -> float:
    """Ionic radius for the oxidation state, falling back to the atomic radius."""
    try:
        value = Species(symbol, state).ionic_radius
        if value:
            return float(value)
    except (ValueError, KeyError):
        pass
    return float(Element(symbol).atomic_radius or 1.5)


def site_map(candidate: str, prototype: str) -> tuple[dict[str, str], dict[str, int], dict[str, int]] | None:
    """Map prototype elements to candidate elements, or None if no sensible match exists."""
    cand_amounts = filters.integer_amounts(candidate)
    proto_amounts = filters.integer_amounts(prototype)
    _, _, cand_ox = filters.charge_assignment(candidate)
    _, _, proto_ox = filters.charge_assignment(prototype)
    if cand_ox is None or proto_ox is None:
        return None

    def key(symbol: str, amounts: dict[str, int], ox: dict[str, int]) -> tuple:
        return (amounts[symbol], ox[symbol] > 0, Element(symbol).X)

    cand = sorted(cand_amounts, key=lambda s: key(s, cand_amounts, cand_ox))
    proto = sorted(proto_amounts, key=lambda s: key(s, proto_amounts, proto_ox))
    mapping = dict(zip(proto, cand))
    for p, c in mapping.items():
        if proto_amounts[p] != cand_amounts[c] or (proto_ox[p] > 0) != (cand_ox[c] > 0):
            return None
    return mapping, proto_ox, cand_ox


def unwrap_slab(frac_z: np.ndarray) -> np.ndarray:
    """Shift fractional z so the largest empty gap sits at the cell boundary."""
    z = np.sort(frac_z % 1.0)
    gaps = np.diff(np.concatenate([z, [z[0] + 1.0]]))
    cut = z[(np.argmax(gaps) + 1) % len(z)]
    return (frac_z - cut) % 1.0


def build(entry: dict, mapping: dict[str, str], proto_ox: dict, cand_ox: dict) -> tuple[Structure, float]:
    attrs = entry["attributes"]
    lattice = np.array(attrs["lattice_vectors"], dtype=float)
    cart = np.array(attrs["cartesian_site_positions"], dtype=float)
    species = attrs["species_at_sites"]
    species = [re.sub(r"[^A-Za-z]", "", s) for s in species]

    dims = attrs.get("dimension_types") or [1, 1, 0]
    vac_axis = dims.index(0) if 0 in dims else 2
    inplane = [i for i in range(3) if i != vac_axis]
    a, b = lattice[inplane[0]], lattice[inplane[1]]
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)

    # height along the normal, unwrapped so the slab is contiguous
    if all(v is not None for v in lattice[vac_axis]) and np.linalg.norm(lattice[vac_axis]) > 0:
        frac = np.linalg.solve(lattice.T, cart.T).T
        frac[:, vac_axis] = unwrap_slab(frac[:, vac_axis])
        cart = frac @ lattice
    heights = cart @ normal
    inplane_cart = cart - np.outer(heights, normal)

    r_proto = sum(radius(s, proto_ox[s]) for s in species)
    r_cand = sum(radius(mapping[s], cand_ox[mapping[s]]) for s in species)
    scale = r_cand / r_proto

    heights = (heights - heights.min()) * scale
    thickness = float(heights.max())
    c = normal * (thickness + VACUUM)
    new_lattice = Lattice(np.array([a * scale, b * scale, c]))
    new_cart = inplane_cart * scale + np.outer(heights + VACUUM / 2, normal)
    structure = Structure(new_lattice, [mapping[s] for s in species], new_cart, coords_are_cartesian=True)
    return structure.get_sorted_structure(), thickness


def mismatch(mapping: dict[str, str], proto_ox: dict, cand_ox: dict) -> float:
    """Oxidation-state change plus relative radius change, summed over elements."""
    total = 0.0
    for p, c in mapping.items():
        rp, rc = radius(p, proto_ox[p]), radius(c, cand_ox[c])
        total += abs(proto_ox[p] - cand_ox[c]) + abs(rc - rp) / rp
    return total


def checks(structure: Structure) -> dict[str, Any]:
    dist = structure.distance_matrix
    np.fill_diagonal(dist, np.inf)
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        spacegroup = f"{sga.get_space_group_symbol()} ({sga.get_space_group_number()})"
    except Exception:  # spglib can fail on odd cells
        spacegroup = "unknown"
    return {"min_distance_A": round(float(dist.min()), 3), "spacegroup_symprec0.1": spacegroup, "nsites": len(structure)}


def prototype_options(client: Any, formula: str, group: str, min_distance: float) -> list[dict]:
    """Every usable prototype for the formula, built and checked, not yet ranked."""
    options = []
    for source, (base_url, citation, licence) in SOURCES.items():
        for entry in fetch_prototypes(client, base_url, filters.optimade_anonymous(formula)):
            proto_formula = entry["attributes"].get("chemical_formula_reduced") or ""
            matched = site_map(formula, proto_formula)
            if matched is None:
                continue
            mapping, proto_ox, cand_ox = matched
            # references keep their own structure, so only exact matches count for them
            exact = filters.optimade_formula(proto_formula) == filters.optimade_formula(formula)
            if group == "reference" and not exact:
                continue
            try:
                structure, thickness = build(entry, mapping, proto_ox, cand_ox)
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                continue
            info = checks(structure)
            if info["min_distance_A"] < min_distance:
                continue
            options.append({
                "structure": structure,
                "score": 0.0 if exact else mismatch(mapping, proto_ox, cand_ox),
                "source": source, "id": str(entry["id"]), "prototype_formula": proto_formula,
                "mapping": mapping, "prototype_ox": proto_ox, "candidate_ox": cand_ox,
                "thickness_A": round(thickness, 3), "citation": citation, "licence": licence, **info,
            })
    return options


def distinct(options: list[dict]) -> list[dict]:
    """Rank by mismatch; the same prototype from two databases is kept once per spacegroup and size."""
    options = sorted(options, key=lambda o: (o["score"], o["nsites"]))
    seen, kept = set(), []
    for option in options:
        tag = (option["spacegroup_symprec0.1"], option["nsites"])
        if tag not in seen:
            seen.add(tag)
            kept.append(option)
    return kept


def mixed_anion_parent(formula: str) -> tuple[str, str, str, float] | None:
    """For one cation and two anions (MnBr2Cl): the single-anion parent (MnBr3), both anions, minor share."""
    amounts = filters.integer_amounts(formula)
    _, _, ox = filters.charge_assignment(formula)
    if ox is None:
        return None
    cations = [s for s in amounts if ox[s] > 0]
    anions = sorted((s for s in amounts if ox[s] < 0), key=lambda s: (-amounts[s], s))
    if len(cations) != 1 or len(anions) != 2 or ox[anions[0]] != ox[anions[1]]:
        return None
    major, minor = anions
    total = amounts[major] + amounts[minor]
    count = amounts[cations[0]]
    parent = f"{cations[0]}{count if count != 1 else ''}{major}{total}"
    return parent, major, minor, amounts[minor] / total


def anion_orderings(base: dict, major: str, minor: str, share: float, min_distance: float) -> list[dict]:
    """Every symmetry-distinct way to put the minor anion on the parent's anion sites, in the parent cell."""
    from itertools import combinations

    from pymatgen.analysis.structure_matcher import StructureMatcher

    parent = base["structure"]
    sites = [i for i, site in enumerate(parent) if site.specie.symbol == major]
    k = share * len(sites)
    if abs(k - round(k)) > 1e-6:
        return []  # needs a supercell; not handled here
    candidates = []
    for chosen in combinations(sites, int(round(k))):
        structure = parent.copy()
        for i in chosen:
            structure.replace(i, minor)
        candidates.append(structure.get_sorted_structure())
    groups = StructureMatcher(primitive_cell=False, scale=False).group_structures(candidates)
    options = []
    for j, group_ in enumerate(groups):
        structure = group_[0]
        info = checks(structure)
        if info["min_distance_A"] < min_distance:
            continue
        option = {key: value for key, value in base.items() if key not in ("structure", "nsites", "min_distance_A",
                                                                          "spacegroup_symprec0.1")}
        option.update(structure=structure, source=f"{base['source']}-ordered", id=f"{base['id']}-o{j + 1}",
                      ordering_degeneracy=len(group_), parent_formula=filters.plain_formula(parent.formula), **info)
        options.append(option)
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", type=Path, default=ROOT / "dft" / "selection.json")
    parser.add_argument("--out", type=Path, default=ROOT / "dft" / "structures")
    parser.add_argument("--cache", type=Path, default=ROOT / "dft" / "cache" / "optimade")
    parser.add_argument("--keep", type=int, default=3, help="prototypes written per candidate")
    parser.add_argument("--max-orderings", type=int, default=8, help="anion orderings written per mixed-anion candidate")
    parser.add_argument("--min-distance", type=float, default=1.6, help="drop structures with closer atoms (A)")
    parser.add_argument("--groups", default="main,reference,control", help="selection groups to build, comma-separated")
    args = parser.parse_args()

    args.out = args.out.resolve()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    control = selection.get("control", {})
    groups = {
        "main": selection.get("main", []),
        "reference": selection.get("reference", []),
        # control compositions are substituted onto prototypes like the main batch
        "control": control.get("picked", []) + control.get("fixed", []),
    }
    formulas = [(f, g) for g in args.groups.split(",") for f in groups[g]]
    client = filters.CachedClient(args.cache)
    rows = []
    for formula, group in formulas:
        anonymous = filters.optimade_anonymous(formula)
        parent = mixed_anion_parent(formula)
        if parent is None:
            options = prototype_options(client, formula, group, args.min_distance)
            kept = distinct(options)[: args.keep]
        else:
            # the database prototypes for A2BC are chalcogen-halides of other kinds; build on the MX3 layer instead
            parent_formula, major, minor, share = parent
            options = []
            for base in distinct(prototype_options(client, parent_formula, group, args.min_distance))[:2]:
                options += anion_orderings(base, major, minor, share, args.min_distance)
            kept = options[: args.max_orderings]
        if not kept:
            rows.append({"formula": formula, "group": group, "anonymous": anonymous, "rank": None})
        for rank, option in enumerate(kept, 1):
            folder = args.out / formula / f"{rank}_{option['source']}_{re.sub(r'[^A-Za-z0-9-]', '_', option['id'])}"
            folder.mkdir(parents=True, exist_ok=True)
            option["structure"].to(filename=str(folder / "POSCAR"), fmt="poscar")
            meta = {k: v for k, v in option.items() if k != "structure"}
            meta.update(formula=formula, group=group, anonymous=anonymous, rank=rank, vacuum_A=VACUUM,
                        note="Unrelaxed starting structure; needs MLIP pre-relax and a VESTA check.")
            (folder / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append({"formula": formula, "group": group, "anonymous": anonymous, "rank": rank,
                         **{k: meta[k] for k in ("source", "id", "prototype_formula", "score", "spacegroup_symprec0.1",
                                                 "nsites", "min_distance_A", "thickness_A")},
                         "folder": str(folder.relative_to(ROOT))})
        print(f"{formula:10s} {anonymous:7s} {len(options):3d} usable prototypes, {len(kept)} written")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
