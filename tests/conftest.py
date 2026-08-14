from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str) -> ModuleType:
    """Import a numerically prefixed pipeline script for unit testing."""
    path = REPO_ROOT / "scripts" / filename
    module_name = f"tested_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def prepare_module() -> ModuleType:
    return load_script("02_prepare_data.py")


@pytest.fixture(scope="session")
def train_module() -> ModuleType:
    return load_script("03_train_model.py")


@pytest.fixture(scope="session")
def discover_module() -> ModuleType:
    return load_script("04_discover.py")
