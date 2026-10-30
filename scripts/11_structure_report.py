"""Stage P hand-off: VESTA files and a PDF of every relaxed prototype, for the check before TRUBA.

For each batch, every POSCAR.mace from 08_mlip_rank.py is copied as a .vasp file (VESTA opens it
directly), named by its MACE rank, whether it goes to VASP and its space group after the MACE
relax. The PDF has a summary page and one page per composition: top and side views and the
numbers from meta.json, the candidate table and the MACE ranking. The monolayer check from 08 is
recomputed here, so batch 1 (ranked before the check existed) is held to the same rule.

Text in the output is Turkish, for Ata and Yeşim Hoca.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.visualize.plot import plot_atoms
from matplotlib.backends.backend_pdf import PdfPages
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
GROUP_TR = {"main": "Ana aday", "control": "Kontrol", "reference": "Referans"}
SOURCE_TR = {"mc2d": "MC2D", "twodmatpedia": "2DMatPedia"}


def layer_check(atoms, bond_scale: float = 1.2) -> int:
    """Bonded pieces in the cell, as in 08_mlip_rank.py (kept separate: 08 needs MACE to import)."""
    i, j = neighbor_list("ij", atoms, [bond_scale * r for r in natural_cutoffs(atoms)])
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms), len(atoms)))
    return int(connected_components(graph, directed=False)[0])


def thickness(atoms) -> float:
    a, b = atoms.cell[0], atoms.cell[1]
    normal = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    heights = atoms.positions @ normal
    return float(heights.max() - heights.min())


def spacegroup(atoms) -> tuple[str, int]:
    try:
        sga = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms), symprec=0.1)
        return sga.get_space_group_symbol(), sga.get_space_group_number()
    except Exception:  # spglib can fail on odd cells
        return "?", 0


def collect(label: str, structures: Path, candidates: Path, scores: Path) -> pd.DataFrame:
    ranking = pd.read_csv(structures / "summary_mlip.csv")
    table = pd.read_csv(candidates).drop_duplicates("formula").set_index("formula")
    # the control picks are not in the candidate table; their score comes from the full scoring, by composition
    all_scores = pd.read_csv(scores) if scores.exists() else pd.DataFrame(columns=["formula", "magnetic_score"])
    by_composition = {Composition(f).reduced_formula: v for f, v in zip(all_scores["formula"], all_scores["magnetic_score"])}
    rows = []
    for _, r in ranking.iterrows():
        folder = structures / r["formula"] / r["folder"]
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        atoms = read(folder / "POSCAR.mace", format="vasp")
        symbol, number = spacegroup(atoms)
        pieces = layer_check(atoms)
        ratio = thickness(atoms) / meta["start_thickness_A"]
        cand = table.loc[r["formula"]] if r["formula"] in table.index else None
        rows.append({
            "batch": label, "formula": r["formula"], "group": r["group"], "folder": folder,
            "mace_rank": int(r["mace_rank"]), "delta_meV": float(r["delta_meV_per_atom"]),
            "converged": bool(r["mace_converged"]), "steps": int(r["mace_steps"]),
            "sg_start": meta["spacegroup_symprec0.1"], "sg_mace": f"{symbol} ({number})",
            "sg_file": symbol.replace("/", "_"), "nsites": len(atoms),
            "a": float(atoms.cell.lengths()[0]), "b": float(atoms.cell.lengths()[1]),
            "gamma": float(atoms.cell.angles()[2]),
            "t_start": meta["start_thickness_A"], "t_mace": thickness(atoms), "t_ratio": ratio,
            "pieces": pieces, "dmin": float(r["mace_min_distance_A"]),
            "holds": pieces == 1 and 0.7 <= ratio <= 1.3,
            "to_vasp_08": bool(r["to_vasp"]),
            "prototype": meta["prototype_formula"], "source": SOURCE_TR.get(meta["source"], meta["source"]),
            "proto_id": meta["id"], "citation": meta["citation"], "mismatch": float(meta["score"]),
            "ox": ", ".join(f"{el}{v:+d}" for el, v in sorted(meta["candidate_ox"].items())),
            "score": float(cand["magnetic_score"]) if cand is not None and pd.notna(cand.get("magnetic_score"))
            else by_composition.get(Composition(r["formula"]).reduced_formula),
            "hull": float(cand["alexandria_hull_distance"])
            if cand is not None and pd.notna(cand.get("alexandria_hull_distance")) else None,
        })
    df = pd.DataFrame(rows)
    # the same rule as 08: the lowest held prototype goes, and the next held one within 25 meV/atom
    df["to_vasp"] = False
    for _, part in df[df["holds"]].sort_values("delta_meV").groupby("formula"):
        e0 = part["delta_meV"].iloc[0]
        df.loc[part.index[0], "to_vasp"] = True
        if len(part) > 1 and part["delta_meV"].iloc[1] - e0 <= 25:
            df.loc[part.index[1], "to_vasp"] = True
    return df


def file_name(row) -> str:
    tag = "VASP" if row["to_vasp"] else ("elendi" if not row["holds"] else "yedek")
    return f"{row['formula']}_{row['mace_rank']}_{tag}_{row['sg_file']}.vasp"


def draw(ax, atoms, rotation: str, repeat: tuple[int, int, int]) -> None:
    plot_atoms(atoms.repeat(repeat), ax, rotation=rotation, radii=0.45, show_unit_cell=0)
    ax.set_axis_off()


def legend(ax, atoms) -> None:
    for el in sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers[s]):
        ax.scatter([], [], s=60, color=jmol_colors[atomic_numbers[el]], edgecolor="k", label=el)
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=False, fontsize=8, handletextpad=0.2, borderaxespad=0,
              ncol=4, columnspacing=1.0)


def fmt(value, spec: str, missing: str = "–") -> str:
    return missing if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


def summary_pages(pdf: PdfPages, df: pd.DataFrame, title: str, per_page: int = 22) -> None:
    rows = []
    for _, r in df[df["to_vasp"]].sort_values(["batch", "group", "formula", "mace_rank"]).iterrows():
        rows.append([r["batch"], GROUP_TR[r["group"]], r["formula"], fmt(r["score"], ".3f"), r["ox"],
                     r["prototype"], r["source"], r["sg_mace"], f"{r['a']:.2f} / {r['b']:.2f}",
                     f"{r['t_mace']:.2f}", str(r["nsites"]), f"{r['delta_meV']:.1f}"])
    cols = ["Parti", "Grup", "Bileşim", "ML skoru", "Yükseltgenme", "Prototip", "Kaynak", "Uzay grubu",
            "a / b (Å)", "Kalınlık (Å)", "Atom", "ΔE (meV/atom)"]
    widths = [0.06, 0.07, 0.09, 0.06, 0.14, 0.15, 0.08, 0.10, 0.09, 0.07, 0.04, 0.08]
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    for n, chunk in enumerate(pages, 1):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.04, 0.95, f"{title}  ({n} / {len(pages)})", fontsize=15, weight="bold")
        fig.text(0.04, 0.915, "Her bileşim için VASP'a giden prototip(ler). Yapılar MACE (makine öğrenmesi potansiyeli) "
                 "ile ön gevşetildi; ΔE yalnız prototip seçimi içindir, sonuç değildir.", fontsize=9)
        fig.text(0.04, 0.893, "ML skoru: parti 1 v3 modeli, parti 2 v4 modeli (yalnız C2DB). Referanslar ve MnBr2 bilinen "
                 "malzemeler, skorları yok.", fontsize=9)
        ax = fig.add_axes([0.03, 0.04, 0.94, 0.83])
        ax.set_axis_off()
        tab = ax.table(cellText=chunk, colLabels=cols, colWidths=widths, loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)
        tab.scale(1, 1.35)
        pdf.savefig(fig)
        plt.close(fig)


def formula_page(pdf: PdfPages, part: pd.DataFrame) -> None:
    first = part.iloc[0]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.965, f"{first['formula']}  ·  {GROUP_TR[first['group']]}  ·  {first['batch']}",
             fontsize=15, weight="bold")
    fig.text(0.06, 0.945, f"ML skoru {fmt(first['score'], '.3f')}  ·  yükseltgenme {first['ox']}  ·  "
             f"Alexandria bulk hull uzaklığı {fmt(first['hull'], '.3f')} eV/atom", fontsize=9)
    height = 0.9 / 3
    for k, (_, r) in enumerate(part.sort_values("mace_rank").iterrows()):
        atoms = read(r["folder"] / "POSCAR.mace", format="vasp")
        top = 0.93 - (k + 1) * height
        ax1 = fig.add_axes([0.04, top + 0.02, 0.30, height - 0.04])
        draw(ax1, atoms, "", (3, 3, 1))
        ax1.set_title("Üstten (3×3)", fontsize=8)
        ax2 = fig.add_axes([0.36, top + 0.02, 0.22, height - 0.04])
        draw(ax2, atoms, "-90x", (3, 1, 1))
        ax2.set_title("Yandan (3×1)", fontsize=8)
        verdict = "VASP'a gider" if r["to_vasp"] else ("Elendi: tek katman tutmadı" if not r["holds"] else "Yedek")
        lines = [
            f"{r['mace_rank']}. sıra  ·  {verdict}",
            f"Dosya: {file_name(r)}",
            f"Prototip: {r['prototype']} ({r['source']}, {r['proto_id'][:13]})",
            f"Uzay grubu: başta {r['sg_start']}, MACE sonrası {r['sg_mace']}",
            f"a = {r['a']:.3f} Å, b = {r['b']:.3f} Å, γ = {r['gamma']:.1f}°, {r['nsites']} atom",
            f"Kalınlık: {r['t_start']:.2f} → {r['t_mace']:.2f} Å (oran {r['t_ratio']:.2f})",
            f"Bağlı parça sayısı: {r['pieces']}  ·  en kısa mesafe {r['dmin']:.2f} Å",
            f"ΔE (en düşük prototipe göre): {r['delta_meV']:.1f} meV/atom",
            f"MACE: {'yakınsadı' if r['converged'] else 'YAKINSAMADI'}, {r['steps']} adım",
            f"Yerleştirme uyumsuzluğu: {r['mismatch']:.3f}",
            f"Kaynak: {r['citation']}",
        ]
        fig.text(0.60, top + height - 0.03, "\n".join(lines), fontsize=7.5, va="top", linespacing=1.5,
                 wrap=True)
        legend(fig.add_axes([0.60, top + 0.01, 0.35, 0.04], frame_on=False), atoms)
    fig.text(0.06, 0.012, "Vakum 20 Å, c ekseni tabakaya dik. MACE spin düzenini görmez; FM/AFM ve kararlılık "
             "VASP'ta belirlenir.", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    batches = [
        ("Parti 1", ROOT / "dft" / "structures", ROOT / "dft" / "candidates_final.csv",
         ROOT / "results" / "v3" / "all_scores.csv"),
        ("Parti 2", ROOT / "dft" / "batch2" / "structures", ROOT / "dft" / "batch2" / "candidates_final.csv",
         ROOT / "results" / "v4" / "all_scores.csv"),
    ]
    df = pd.concat([collect(*b) for b in batches], ignore_index=True)

    for _, r in df.iterrows():
        sub = {"reference": "referans", "control": "kontrol"}.get(r["group"], r["batch"].lower().replace(" ", ""))
        target = args.out / "yapilar" / sub / r["formula"]
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(r["folder"] / "POSCAR.mace", target / file_name(r))

    args.out.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out / "yapilar.pdf") as pdf:
        summary_pages(pdf, df, "Aşama P: VASP'a gidecek tek katman yapılar")
        for _, part in df.groupby(["batch", "group", "formula"], sort=True):
            formula_page(pdf, part)

    keep = ["batch", "group", "formula", "mace_rank", "to_vasp", "holds", "to_vasp_08", "sg_start", "sg_mace",
            "nsites", "a", "b", "gamma", "t_start", "t_mace", "t_ratio", "pieces", "dmin", "delta_meV",
            "prototype", "source", "proto_id", "score", "hull", "ox"]
    df[keep].round(4).to_csv(args.out / "yapilar.csv", index=False)
    changed = df[df["to_vasp"] != df["to_vasp_08"]]
    print(f"{len(df)} structures, {int(df['to_vasp'].sum())} to VASP; differs from 08's choice: "
          f"{', '.join(changed['formula'] + '/' + changed['mace_rank'].astype(str)) or 'none'}")


if __name__ == "__main__":
    main()
