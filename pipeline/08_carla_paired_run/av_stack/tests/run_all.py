"""Run every test without pytest:  python tests/run_all.py"""
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

failed = 0
for mod_name in ("test_separation", "test_planner", "test_lqr", "test_mpc"):
    mod = importlib.import_module(mod_name)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS {mod_name}.{name} ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {mod_name}.{name}: {exc!r}")
print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
