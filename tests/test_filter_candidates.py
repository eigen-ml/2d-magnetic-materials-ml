from __future__ import annotations

import pandas as pd
import pytest


def test_optimade_formula_spellings(filter_module):
    assert filter_module.optimade_formula("Mn2CoCl4") == "Cl4CoMn2"
    assert filter_module.optimade_anonymous("Mn2CrS4") == "A4B2C"
    assert filter_module.optimade_anonymous("CrI3") == "A3B"
    assert filter_module.plain_formula("Mn(CrCl2)2") == "MnCr2Cl4"


@pytest.mark.parametrize(
    ("formula", "balanced", "ordered"),
    [
        ("Mn2CrBr4", False, False),  # Br -1 x4 leaves +4 for three metals
        ("Mn2FeO4", True, True),  # Mn+3, Fe+2
        ("CrI3", True, True),
        ("MoS2", True, True),
    ],
)
def test_charge_assignment(filter_module, formula, balanced, ordered):
    is_balanced, is_ordered, assignment = filter_module.charge_assignment(formula)
    assert (is_balanced, is_ordered) == (balanced, ordered)
    if balanced:
        amounts = filter_module.integer_amounts(formula)
        assert sum(assignment[s] * n for s, n in amounts.items()) == 0


def test_formula_variants_put_cations_in_every_order(filter_module):
    _, _, assignment = filter_module.charge_assignment("Mn2FeO4")
    assert set(filter_module.formula_variants("Mn2FeO4", assignment)) == {"FeMn2O4", "Mn2FeO4"}


def test_two_d_word_needs_the_formula_as_a_token(filter_module):
    variants = ["CrI3"]
    assert filter_module.has_formula_and_2d_word("Monolayer CrI3 is a ferromagnet", variants)
    assert not filter_module.has_formula_and_2d_word("Bulk CrI3 crystals", variants)
    assert not filter_module.has_formula_and_2d_word("Monolayer CrI35 alloy", variants)
    assert filter_module.has_formula_and_2d_word("2D Cr<sub></sub>I3 sheets", variants)


def fake_fetch(answers):
    calls = []

    def fetch(url, params):
        calls.append((url, params))
        for key, value in answers.items():
            if key in url + str(params):
                return value
        return {"data": [], "meta": {"data_returned": 0}}

    return fetch, calls


def test_cached_client_queries_once(filter_module, tmp_path):
    fetch, calls = fake_fetch({})
    client = filter_module.CachedClient(tmp_path, fetch)
    client.get("https://example.org", {"a": 1})
    client.get("https://example.org", {"a": 1})
    assert len(calls) == 1


def test_optimade_matches_follows_pagination(filter_module, tmp_path):
    first = {
        "data": [{"attributes": {"chemical_formula_reduced": "CrI3"}}],
        "links": {"next": "https://db.test/v1/structures?page=2"},
    }
    second = {"data": [{"attributes": {"chemical_formula_reduced": "Fe2O3"}}], "links": {}}
    fetch, _ = fake_fetch({"page=2": second, "filter": first})
    client = filter_module.CachedClient(tmp_path, fetch)
    found, ok = filter_module.optimade_matches(
        client, "https://db.test", "chemical_formula_reduced", ["CrI3", "Fe2O3", "MoS2"], True
    )
    assert ok and found == {"CrI3", "Fe2O3"}


def test_optimade_matches_reports_a_dead_provider(filter_module, tmp_path):
    fetch, _ = fake_fetch({"filter": "<html>502</html>"})
    client = filter_module.CachedClient(tmp_path, fetch)
    found, ok = filter_module.optimade_matches(
        client, "https://db.test", "chemical_formula_reduced", ["CrI3"], False
    )
    assert found == set() and not ok


def test_literature_check_counts_only_2d_mentions(filter_module, tmp_path):
    openalex = {
        "results": [
            {"id": "W1", "title": "Monolayer CrI3", "publication_year": 2018},
            {"id": "W2", "title": "Bulk CrI3 growth", "publication_year": 1990},
        ]
    }
    fetch, _ = fake_fetch({"openalex": openalex, "arxiv": "<feed></feed>"})
    client = filter_module.CachedClient(tmp_path, fetch)
    result = filter_module.literature_check(client, client, ["CrI3"])
    assert result["literature_hits"] == 2
    assert result["literature_2d_hits"] == 1


def test_select_groups_caps_each_family(filter_module):
    rows = []
    for i, family in enumerate(["F1", "F1", "F1", "F2", "F3"]):
        rows.append(
            {
                "formula": f"X{i}", "formula_plain": f"X{i}", "family": family,
                "magnetic_score": 0.99, "raw_score": 1.0 - i / 100,
                **{name: True for name in ("S1", "S2", "S3", "S4", "S5", "S6", "S8")},
            }
        )
    rows.append(
        {
            "formula": "VCl3", "formula_plain": "VCl3", "family": "F4",
            "magnetic_score": 0.99, "raw_score": 0.5,
            **{name: True for name in ("S1", "S2", "S3", "S4", "S8")}, "S5": True, "S6": False,
        }
    )
    table = filter_module.select_groups(pd.DataFrame(rows))
    groups = dict(zip(table["formula"], table["group"]))
    assert [f for f, g in groups.items() if g == "main"] == ["X0", "X1", "X3", "X4"]
    assert groups["X2"] == "" and groups["VCl3"] == "reference"
    assert table.set_index("formula").loc["VCl3", "first_failure"] == "S6"
