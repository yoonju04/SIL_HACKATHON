import conftest  # noqa: F401  (sets sys.path)
"""The three modules must not import each other (or CARLA)."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ("planner", "mpc_controller", "lqr_controller")
FORBIDDEN_EXTRA = ("carla", "integration", "sim", "steering")


def imported_top_levels(pkg):
    names = set()
    for fn in os.listdir(os.path.join(ROOT, pkg)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(ROOT, pkg, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_modules_are_independent():
    for pkg in MODULES:
        used = imported_top_levels(pkg)
        others = set(MODULES) - {pkg}
        bad = used & (others | set(FORBIDDEN_EXTRA))
        assert not bad, f"{pkg} imports {bad}"
        assert used <= {"numpy", "scipy", "math", "time", "json", "dataclasses", "functools",
                        "collections", "__future__", "osqp"}, f"{pkg}: unexpected deps {used}"


if __name__ == "__main__":
    test_modules_are_independent()
    print("PASS")
