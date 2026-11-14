import pytest
import torch


@pytest.fixture(autouse=True)
def _seeded():
    """Every test builds its random networks from the same seed, so a tolerance holds on every run."""
    torch.manual_seed(0)
