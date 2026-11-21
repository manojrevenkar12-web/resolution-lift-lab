import copy

import pytest

from rll.config import load_world
from rll.world.sim import simulate


def small_config() -> dict:
    cfg = copy.deepcopy(load_world())
    cfg.update(days=42, workspaces=3)
    cfg["resolution"]["model_upgrade"]["day"] = 21
    return cfg


@pytest.fixture(scope="session")
def small_world():
    return simulate(7, small_config())
