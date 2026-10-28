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
        ("MnCoCl3", False, False),  # needs Co+1, 0.8% of cobalt's ICSD entries
    ],
)
def test_charge_assignment(filter_module, formula, balanced, ordered):
    pytest.importorskip("smact")
    is_balanced, is_ordered, assignment = filter_module.charge_assignment(formula)
    assert (is_balanced, is_ordered) == (balanced, ordered)
    if balanced:
        amounts = filter_module.integer_amounts(formula)
        assert sum(assignment[s] * n for s, n in amounts.items()) == 0


def test_charge_assignment_prefers_the_usual_states(filter_module):
    pytest.importorskip("smact")
    _, _, assignment = filter_module.charge_assignment("MnFeTe3")
    assert assignment == {"Fe": 3, "Mn": 3, "Te": -2}
    assert filter_module.assignment_likelihood(assignment) > filter_module.assignment_likelihood(
        {"Fe": 2, "Mn": 4, "Te": -2}
    )


def test_formula_variants_put_cations_in_every_order(filter_module):
    pytest.importorskip("smact")
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


def test_oxidation_states_drop_the_rare_tail(filter_module):
    pytest.importorskip("smact")
    assert 1 not in filter_module.oxidation_states("Co")
    assert {2, 3} <= set(filter_module.oxidation_states("Co"))
    assert 3 in filter_module.oxidation_states("Mn")


def _passing_row(formula, family, cations, raw):
    return {
        "formula": formula, "formula_plain": formula, "family": family, "cation_set": cations,
        "magnetic_score": 0.99, "raw_score": raw,
        **{name: True for name in ("S1", "S2", "S3", "S4", "S5", "S6", "S8")},
    }


def test_select_groups_caps_each_family_and_cation_set(filter_module):
    rows = [
        _passing_row("A", "F1", "Mn+V", 0.99),
        _passing_row("B", "F1", "Co+Mn", 0.98),
        _passing_row("C", "F1", "Fe+Mn", 0.97),  # third in F1
        _passing_row("D", "F2", "Mn+V", 0.96),
        _passing_row("E", "F3", "Mn+V", 0.95),  # third Mn+V
        _passing_row("F", "F4", "Cr+Fe", 0.94),
    ]
    failing = _passing_row("G", "F5", "Cr+Ni", 0.999)
    failing["S6"] = False
    table = filter_module.select_groups(pd.DataFrame(rows + [failing]))
    groups = dict(zip(table["formula"], table["group"]))
    assert [f for f in "ABCDEFG" if groups[f] == "main"] == ["A", "B", "D", "F"]
    assert table.set_index("formula").loc["G", "first_failure"] == "S6"
