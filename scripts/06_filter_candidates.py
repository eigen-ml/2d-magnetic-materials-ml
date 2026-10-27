#!/usr/bin/env python3
"""Filter screened compositions into DFT candidates (rules in dft/CRITERIA.md).

S1 score, S2 in-distribution, S3 charge balance, S4 electronegativity, S5 not in
a 2D database (OPTIMADE), S6 not in the 2D literature (OpenAlex, arXiv), S7 family
diversity, S8 a same-stoichiometry 2D prototype exists. Network answers are cached
under dft/cache/ so a rerun does not query again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, product
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
from pymatgen.core import Composition, Element


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = REPO_ROOT / "results" / "v3" / "screened_candidates.csv"
DEFAULT_TRAINING = REPO_ROOT / "data" / "v3" / "combined_net_moment.csv"
DEFAULT_C2DB = REPO_ROOT / "data" / "legacy_hybrid" / "combined_dataset.csv"
DEFAULT_THESIS = REPO_ROOT / "results" / "legacy" / "novel_candidates.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dft"
DEFAULT_CACHE_DIR = REPO_ROOT / "dft" / "cache"

SCORE_MIN = 0.98
MIN_TEMPLATE_COUNT = 20
FAMILY_LIMIT = 2
MAIN_SIZE = 10
REFERENCE_EXTRA = 2
THESIS_TOP = 10
BENCHMARK = "CrI3"

# name: (base URL, accepts OR filters, kind). "2d" hits fail S5, "bulk" hits are recorded only.
OPTIMADE_PROVIDERS: dict[str, tuple[str, bool, str]] = {
    "mc2d": ("https://optimade.materialscloud.org/main/mc2d", True, "2d"),
    "twodmatpedia": ("http://optimade.2dmatpedia.org", True, "2d"),
    "mp": ("https://optimade.materialsproject.org", True, "bulk"),
    "alexandria": ("https://alexandria.icams.rub.de/pbe", False, "bulk"),
    "jarvis": ("https://jarvis.nist.gov/optimade/jarvisdft", False, "bulk"),
}
TWO_D_WORDS = re.compile(
    r"monolayer|two-dimensional|\b2D\b|single-layer|single layer|nanosheet|few-layer|exfoliat",
    re.IGNORECASE,
)
FILTERS = ("S1", "S2", "S3", "S4", "S5", "S6", "S8")

Fetch = Callable[[str, dict[str, Any]], dict[str, Any] | str]


class FilterError(RuntimeError):
    """Raised when filter inputs are missing or unusable."""


# --- composition helpers -----------------------------------------------------


def integer_amounts(formula: str) -> dict[str, int]:
    """Return the reduced composition as element -> integer count."""
    reduced = Composition(formula).reduced_composition
    return {str(element): int(round(amount)) for element, amount in reduced.items()}


def plain_formula(formula: str) -> str:
    """Write a formula without the brackets pymatgen uses, cations first by electronegativity."""
    amounts = integer_amounts(formula)
    ordered = sorted(amounts, key=lambda symbol: (Element(symbol).X, symbol))
    return "".join(f"{symbol}{amounts[symbol] if amounts[symbol] != 1 else ''}" for symbol in ordered)


def optimade_formula(formula: str) -> str:
    """OPTIMADE chemical_formula_reduced: elements in alphabetical order."""
    amounts = integer_amounts(formula)
    return "".join(
        f"{symbol}{amounts[symbol] if amounts[symbol] != 1 else ''}" for symbol in sorted(amounts)
    )


def optimade_anonymous(formula: str) -> str:
    """OPTIMADE chemical_formula_anonymous: counts in decreasing order, labelled A, B, ..."""
    counts = sorted(integer_amounts(formula).values(), reverse=True)
    letters = "ABCDEFGHIJ"
    return "".join(f"{letters[i]}{n if n != 1 else ''}" for i, n in enumerate(counts))


def oxidation_states(symbol: str) -> tuple[int, ...]:
    """ICSD oxidation states, or the common ones if pymatgen has no ICSD list."""
    element = Element(symbol)
    return tuple(element.icsd_oxidation_states) or tuple(element.common_oxidation_states)


def charge_assignment(formula: str) -> tuple[bool, bool, dict[str, int] | None]:
    """Return (charge balanced, electronegativity ordered, the assignment used).

    One oxidation state per element (SMACT rule). An electronegativity-ordered
    assignment is preferred; otherwise the first neutral one is returned.
    """
    amounts = integer_amounts(formula)
    symbols = sorted(amounts)
    first_neutral: dict[str, int] | None = None
    for states in product(*(oxidation_states(symbol) for symbol in symbols)):
        if sum(state * amounts[symbol] for symbol, state in zip(symbols, states)) != 0:
            continue
        assignment = dict(zip(symbols, states))
        first_neutral = first_neutral or assignment
        cations = [Element(s).X for s, state in assignment.items() if state > 0]
        anions = [Element(s).X for s, state in assignment.items() if state < 0]
        if anions and (not cations or max(cations) < min(anions)):
            return True, True, assignment
    return first_neutral is not None, False, first_neutral


def anion_set(formula: str, assignment: dict[str, int] | None) -> str:
    """Anions from the assignment, or the most electronegative element if there is none."""
    if assignment:
        anions = sorted(symbol for symbol, state in assignment.items() if state < 0)
        if anions:
            return "+".join(anions)
    return max(integer_amounts(formula), key=lambda symbol: Element(symbol).X)


def formula_variants(formula: str, assignment: dict[str, int] | None, limit: int = 6) -> list[str]:
    """Spellings of the formula with the cations in every order and the anions last."""
    amounts = integer_amounts(formula)
    if assignment:
        cations = [s for s in amounts if assignment.get(s, 0) > 0]
        anions = [s for s in amounts if assignment.get(s, 0) <= 0]
    else:
        most_negative = max(amounts, key=lambda symbol: Element(symbol).X)
        cations = [s for s in amounts if s != most_negative]
        anions = [most_negative]
    anions = sorted(anions, key=lambda symbol: (Element(symbol).X, symbol))
    write = lambda symbols: "".join(  # noqa: E731
        f"{s}{amounts[s] if amounts[s] != 1 else ''}" for s in symbols
    )
    variants: list[str] = []
    for order in permutations(sorted(cations)):
        spelled = write(list(order) + anions)
        if spelled not in variants:
            variants.append(spelled)
        if len(variants) >= limit:
            break
    return variants or [plain_formula(formula)]


def has_formula_and_2d_word(text: str, variants: Sequence[str]) -> bool:
    """True if the text names one of the formula spellings and a 2D word."""
    cleaned = re.sub(r"<[^>]+>", "", text or "")
    named = any(re.search(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![a-z0-9])", cleaned) for v in variants)
    return named and bool(TWO_D_WORDS.search(cleaned))


# --- cached network access ---------------------------------------------------


def default_fetch(url: str, params: dict[str, Any]) -> dict[str, Any] | str:
    """GET a URL and return JSON, or the raw text if the body is not JSON."""
    import requests

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code >= 500 and attempt < 2:
                time.sleep(2 + 3 * attempt)
                continue
            try:
                return response.json()
            except ValueError:
                return response.text
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 + 3 * attempt)
    return ""


class CachedClient:
    """Wrap a fetch function with an on-disk JSON cache keyed by URL and parameters."""

    def __init__(self, cache_dir: Path | None, fetch: Fetch = default_fetch, delay: float = 0.0):
        self.cache_dir = cache_dir
        self.fetch = fetch
        self.delay = delay
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, params: dict[str, Any]) -> dict[str, Any] | str:
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json" if self.cache_dir is not None else None
        if path is not None and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        response = self.fetch(url, params)
        if path is not None:
            record = {
                "url": url,
                "params": params,
                "fetched": dt.date.today().isoformat(),
                "response": response,
            }
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        if self.delay:
            time.sleep(self.delay)
        return response


def optimade_matches(
    client: CachedClient,
    base_url: str,
    field: str,
    values: Sequence[str],
    supports_or: bool,
    batch: int = 40,
) -> tuple[set[str], bool]:
    """Return the values that have at least one OPTIMADE structure, and whether every query answered."""
    found: set[str] = set()
    healthy = True
    groups = [values[i : i + batch] for i in range(0, len(values), batch)] if supports_or else [
        [value] for value in values
    ]

    def run(group: Sequence[str]) -> tuple[set[str], bool]:
        query = " OR ".join(f'{field}="{value}"' for value in group)
        single = len(group) == 1
        params = {"filter": query, "response_fields": field, "page_limit": 1 if single else 100}
        response = client.get(base_url.rstrip("/") + "/v1/structures", params)
        hits: set[str] = set()
        for _ in range(50):  # follow pagination; an OR query can return many entries
            if not isinstance(response, dict) or "data" not in response:
                return set(), False
            hits |= {item.get("attributes", {}).get(field) for item in response["data"]}
            if single:
                if response.get("meta", {}).get("data_returned", len(response["data"])):
                    hits.add(group[0])
                break
            next_link = response.get("links", {}).get("next")
            if isinstance(next_link, dict):
                next_link = next_link.get("href")
            if not next_link:
                break
            response = client.get(next_link, {})
        return {hit for hit in hits if hit in group}, True

    with ThreadPoolExecutor(max_workers=1 if supports_or else 4) as pool:
        for hits, ok in pool.map(run, groups):
            found |= hits
            healthy &= ok
    return found, healthy


def openalex_works(client: CachedClient, variant: str) -> list[dict[str, str]]:
    """Works whose title or abstract mentions the variant, with reconstructed abstracts."""
    params = {
        "filter": f'title_and_abstract.search:"{variant}"',
        "per_page": 50,
        "select": "id,doi,title,publication_year,abstract_inverted_index",
    }
    response = client.get("https://api.openalex.org/works", params)
    if not isinstance(response, dict):
        return []
    works = []
    for item in response.get("results", []):
        index = item.get("abstract_inverted_index") or {}
        positions = sorted((pos, word) for word, places in index.items() for pos in places)
        works.append(
            {
                "source": "openalex",
                "id": item.get("doi") or item.get("id") or "",
                "year": str(item.get("publication_year") or ""),
                "text": f"{item.get('title') or ''} {' '.join(word for _, word in positions)}",
                "title": item.get("title") or "",
            }
        )
    return works


def arxiv_works(client: CachedClient, variants: Sequence[str]) -> list[dict[str, str]]:
    """arXiv entries matching any variant (one OR query; arXiv asks for 3 s between calls)."""
    query = " OR ".join(f'all:"{variant}"' for variant in variants)
    response = client.get(
        "http://export.arxiv.org/api/query", {"search_query": query, "max_results": 50}
    )
    text = response if isinstance(response, str) else ""
    works = []
    for entry in re.findall(r"<entry>(.*?)</entry>", text, flags=re.DOTALL):
        title = re.search(r"<title>(.*?)</title>", entry, flags=re.DOTALL)
        summary = re.search(r"<summary>(.*?)</summary>", entry, flags=re.DOTALL)
        ident = re.search(r"<id>(.*?)</id>", entry)
        published = re.search(r"<published>(\d{4})", entry)
        works.append(
            {
                "source": "arxiv",
                "id": ident.group(1) if ident else "",
                "year": published.group(1) if published else "",
                "text": f"{title.group(1) if title else ''} {summary.group(1) if summary else ''}",
                "title": " ".join((title.group(1) if title else "").split()),
            }
        )
    return works


def literature_check(
    openalex: CachedClient,
    arxiv: CachedClient,
    variants: Sequence[str],
) -> dict[str, Any]:
    """Count works naming the formula, and those that also use a 2D word."""
    works = [work for variant in variants for work in openalex_works(openalex, variant)]
    works += arxiv_works(arxiv, variants)
    named = [
        work
        for work in works
        if any(re.search(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![a-z0-9])", work["text"]) for v in variants)
    ]
    two_d = [work for work in named if has_formula_and_2d_word(work["text"], variants)]
    seen: set[str] = set()
    examples = []
    for work in two_d + named:
        if work["id"] in seen:
            continue
        seen.add(work["id"])
        examples.append(f"{work['year']} {work['title'][:90]} [{work['id']}]")
        if len(examples) == 3:
            break
    return {
        "literature_hits": len({work["id"] for work in named}),
        "literature_2d_hits": len({work["id"] for work in two_d}),
        "literature_examples": " | ".join(examples),
    }


# --- filtering ---------------------------------------------------------------


def template_counts(training: pd.DataFrame) -> Counter:
    """Count training compositions per OPTIMADE anonymous formula."""
    unique = training["formula_normalized"].dropna().astype(str).unique()
    return Counter(optimade_anonymous(formula) for formula in unique)


def local_filters(
    candidates: pd.DataFrame,
    templates: Counter,
    c2db_formulas: set[str],
) -> pd.DataFrame:
    """Apply S1-S4 and the local part of S5 to every candidate."""
    rows = []
    for record in candidates.to_dict("records"):
        formula = record["formula"]
        balanced, ordered, assignment = charge_assignment(formula)
        anonymous = optimade_anonymous(formula)
        rows.append(
            {
                **record,
                "formula_plain": plain_formula(formula),
                "optimade_formula": optimade_formula(formula),
                "anonymous_formula": anonymous,
                "family": f"{anonymous}:{anion_set(formula, assignment)}",
                "oxidation_states": " ".join(f"{s}{v:+d}" for s, v in sorted((assignment or {}).items())),
                "training_template_count": int(templates.get(anonymous, 0)),
                "S1": record["magnetic_score"] >= SCORE_MIN,
                "S2": templates.get(anonymous, 0) >= MIN_TEMPLATE_COUNT,
                "S3": balanced,
                "S4": ordered,
                "in_c2db": Composition(formula).reduced_formula in c2db_formulas,
            }
        )
    return pd.DataFrame(rows)


def network_filters(
    table: pd.DataFrame,
    optimade: CachedClient,
    openalex: CachedClient,
    arxiv: CachedClient,
    evaluate: pd.Series,
    providers: dict[str, tuple[str, bool, str]] = OPTIMADE_PROVIDERS,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Apply S5, S6 and S8 to the rows selected by ``evaluate``."""
    table = table.copy()
    subset = table.loc[evaluate]
    formulas = sorted(set(subset["optimade_formula"]))
    provider_ok: dict[str, bool] = {}
    for name, (base_url, supports_or, kind) in providers.items():
        found, healthy = optimade_matches(
            optimade, base_url, "chemical_formula_reduced", formulas, supports_or
        )
        provider_ok[name] = healthy
        table[f"db_{name}"] = pd.NA
        table.loc[evaluate, f"db_{name}"] = subset["optimade_formula"].isin(found)

    two_d = [f"db_{n}" for n, (_, _, kind) in providers.items() if kind == "2d"]
    bulk = [f"db_{n}" for n, (_, _, kind) in providers.items() if kind == "bulk"]
    table["S5"] = pd.NA
    table.loc[evaluate, "S5"] = ~(
        table.loc[evaluate, two_d].astype(bool).any(axis=1) | table.loc[evaluate, "in_c2db"]
    )
    table["bulk_known"] = pd.NA
    table.loc[evaluate, "bulk_known"] = table.loc[evaluate, bulk].astype(bool).any(axis=1)

    templates = sorted(set(table["anonymous_formula"]))
    prototypes: Counter = Counter()
    for name, (base_url, supports_or, kind) in providers.items():
        if kind != "2d":
            continue
        for template in templates:
            params = {
                "filter": f'chemical_formula_anonymous="{template}"',
                "response_fields": "id",
                "page_limit": 1,
            }
            response = optimade.get(base_url.rstrip("/") + "/v1/structures", params)
            if isinstance(response, dict):
                prototypes[template] += int(response.get("meta", {}).get("data_returned") or 0)
    table["prototype_count_2d"] = table["anonymous_formula"].map(lambda t: prototypes.get(t, 0))
    table["S8"] = table["prototype_count_2d"] > 0

    literature = []
    for record in table.loc[evaluate].to_dict("records"):
        _, _, assignment = charge_assignment(record["formula"])
        result = literature_check(openalex, arxiv, formula_variants(record["formula"], assignment))
        literature.append({"index": record["_row"], **result})
    lit = pd.DataFrame(literature).set_index("index") if literature else pd.DataFrame()
    for column in ("literature_hits", "literature_2d_hits", "literature_examples"):
        table[column] = lit[column].reindex(table["_row"]).to_numpy() if not lit.empty else pd.NA
    table["S6"] = pd.NA
    table.loc[evaluate, "S6"] = table.loc[evaluate, "literature_2d_hits"].astype(int) == 0
    table["literature_bulk"] = pd.NA
    table.loc[evaluate, "literature_bulk"] = (
        table.loc[evaluate, "literature_hits"].astype(int) > 0
    ) & table.loc[evaluate, "S6"].astype(bool)
    return table, provider_ok


def first_failure(row: pd.Series) -> str:
    """The first filter the row fails, 'not evaluated' if it stops there, or 'pass'."""
    for name in FILTERS:
        value = row.get(name)
        if value is pd.NA or value is None or (isinstance(value, float) and pd.isna(value)):
            return "not evaluated"
        if not bool(value):
            return name
    return "pass"


def select_groups(table: pd.DataFrame) -> pd.DataFrame:
    """Pick the main 10 with the family cap (S7) and the extra reference candidates."""
    table = table.sort_values(
        ["magnetic_score", "raw_score", "formula"], ascending=[False, False, True]
    ).copy()
    table["first_failure"] = table.apply(first_failure, axis=1)
    table["S7"] = pd.NA
    table["group"] = ""
    per_family: Counter = Counter()
    main = 0
    for index, row in table.iterrows():
        if row["first_failure"] != "pass":
            continue
        if per_family[row["family"]] >= FAMILY_LIMIT:
            table.at[index, "S7"] = False
            continue
        table.at[index, "S7"] = True
        if main < MAIN_SIZE:
            per_family[row["family"]] += 1
            table.at[index, "group"] = "main"
            main += 1
    reference = 0
    for index, row in table.iterrows():
        if reference >= REFERENCE_EXTRA:
            break
        known_2d = row["first_failure"] in ("S5", "S6")
        if known_2d and bool(row.get("S8")) and row["formula_plain"] != BENCHMARK:
            table.at[index, "group"] = "reference"
            reference += 1
    return table


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--c2db", type=Path, default=DEFAULT_C2DB)
    parser.add_argument("--thesis", type=Path, default=DEFAULT_THESIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Counter, set[str]]:
    """Read the v3 screening, the thesis list, training templates and C2DB formulas."""
    for path in (args.candidates, args.training, args.c2db, args.thesis):
        if not path.is_file():
            raise FilterError(f"input not found: {path}")
    candidates = pd.read_csv(args.candidates)
    missing = {"formula", "magnetic_score", "raw_score"} - set(candidates.columns)
    if missing:
        raise FilterError(f"{args.candidates} lacks column(s): {', '.join(sorted(missing))}")
    thesis = pd.read_csv(args.thesis).sort_values("probability", ascending=False).head(THESIS_TOP)
    hybrid = pd.read_csv(args.c2db)
    c2db = hybrid.loc[hybrid.get("source", pd.Series("C2DB", index=hybrid.index)) == "C2DB"]
    c2db_formulas = {Composition(f).reduced_formula for f in c2db["formula_normalized"].dropna()}
    return candidates, thesis, template_counts(pd.read_csv(args.training)), c2db_formulas


def main(argv: list[str] | None = None) -> int:
    """Run the candidate filter and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        candidates, thesis, templates, c2db_formulas = load_inputs(args)
        candidates = candidates.assign(list="v3")
        thesis_rows = candidates.loc[candidates["formula"].isin(thesis["formula"])].copy()
        missing = thesis.loc[~thesis["formula"].isin(candidates["formula"]), ["formula"]]
        thesis_rows = pd.concat([thesis_rows, missing.assign(magnetic_score=0.0, raw_score=0.0)])
        thesis_rows["list"] = "thesis_top10"
        table = local_filters(pd.concat([candidates, thesis_rows], ignore_index=True), templates, c2db_formulas)
        table["_row"] = range(len(table))
        local_ok = table[["S1", "S2", "S3", "S4"]].all(axis=1)
        evaluate = local_ok | (table["list"] == "thesis_top10")
        optimade = CachedClient(args.cache_dir / "optimade")
        openalex = CachedClient(args.cache_dir / "openalex", delay=0.15)
        arxiv = CachedClient(args.cache_dir / "arxiv", delay=3.0)
        table, provider_ok = network_filters(table, optimade, openalex, arxiv, evaluate)
    except (FilterError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    v3 = select_groups(table.loc[table["list"] == "v3"])
    thesis_table = select_groups(table.loc[table["list"] == "thesis_top10"]).assign(group="thesis_top10")
    columns = [
        "group", "formula", "formula_plain", "magnetic_score", "raw_score", "first_failure",
        "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "oxidation_states", "anonymous_formula",
        "family", "training_template_count", "in_c2db",
        *[f"db_{name}" for name in OPTIMADE_PROVIDERS], "bulk_known", "prototype_count_2d",
        "literature_hits", "literature_2d_hits", "literature_bulk", "literature_examples",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = pd.concat([v3, thesis_table], ignore_index=True)[columns]
    final.to_csv(args.output_dir / "candidates_final.csv", index=False)

    funnel = {"screened": int(len(v3))}
    alive = pd.Series(True, index=v3.index)
    for name in ("S1", "S2", "S3", "S4", "S5", "S6", "S8"):
        alive &= v3[name].astype("boolean").fillna(False).astype(bool)
        funnel[f"after_{name}"] = int(alive.sum())
    summary = {
        "run_date_utc": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "criteria": "dft/CRITERIA.md",
        "funnel": funnel,
        "optimade_providers_answered": provider_ok,
        "main": v3.loc[v3["group"] == "main", "formula_plain"].tolist(),
        "reference": [BENCHMARK, *v3.loc[v3["group"] == "reference", "formula_plain"].tolist()],
        "thesis_top10_first_failure": dict(zip(thesis_table["formula_plain"], thesis_table["first_failure"])),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for step, count in funnel.items():
        print(f"{step}: {count:,}")
    print("Main:", ", ".join(summary["main"]) or "none")
    print("Reference:", ", ".join(summary["reference"]))
    for name, ok in provider_ok.items():
        if not ok:
            print(f"WARNING: OPTIMADE provider {name} did not answer every query", file=sys.stderr)
    print(f"Wrote: {args.output_dir / 'candidates_final.csv'}")
    print(f"Wrote: {args.output_dir / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
