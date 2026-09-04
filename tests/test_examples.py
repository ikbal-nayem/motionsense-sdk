"""Every example must import cleanly against the real API.

Examples are documentation that can rot silently: a renamed method or a changed
signature breaks them without breaking anything else. Importing each one catches
that, without running any of them (they all guard on ``__main__``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


def test_examples_exist():
    assert EXAMPLES, "no examples found"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_imports(path: Path):
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
