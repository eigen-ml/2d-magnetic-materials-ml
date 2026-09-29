import importlib.util
from pathlib import Path

import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "05_grouped_validation.py"
    spec = importlib.util.spec_from_file_location("grouped_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chemical_system_ignores_stoichiometry():
    module = _load()
    X = pd.DataFrame(
        {"elem_Cr": [0.25, 1 / 3, 0.0], "elem_I": [0.75, 2 / 3, 0.5], "elem_Fe": [0.0, 0.0, 0.5], "natoms": [4, 3, 2]}
    )
    assert module.chemical_system(X).tolist() == ["Cr-I", "Cr-I", "Fe-I"]
