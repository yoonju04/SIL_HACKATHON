from __future__ import annotations

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
GARAGE = PROJECT/'third_party'/'carla_garage'


def prepare() -> None:
    for path in (PROJECT/'third_party'/'carla_0915', GARAGE/'Bench2Drive'/'scenario_runner',
                 GARAGE/'Bench2Drive'/'leaderboard', GARAGE/'team_code', PROJECT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
