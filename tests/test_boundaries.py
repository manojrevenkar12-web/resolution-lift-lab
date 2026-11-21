"""The truth stays in the world, and the world does not read the methods' settings."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHOD_MODULES = ["rll/topics.py", "rll/resolution.py", "rll/lift.py"]


def imports(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_methods_never_import_the_world():
    for m in METHOD_MODULES:
        assert not any(i.startswith("rll.world") for i in imports(m)), m
        assert "world.yml" not in (ROOT / m).read_text()


def test_the_world_never_reads_method_settings():
    src = (ROOT / "rll/world/sim.py").read_text()
    assert "methods.yml" not in src and "load_methods" not in src
