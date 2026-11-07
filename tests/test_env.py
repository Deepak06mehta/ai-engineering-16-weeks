"""Environment tests. Skipped cleanly when a dependency is not installed yet."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="run `make setup` first")


@pytest.mark.needs_torch
def test_device_selection_returns_a_real_device():
    from src.common.device import get_device

    assert str(get_device()) in {"mps", "cuda", "cpu"}


@pytest.mark.needs_torch
def test_seeding_is_reproducible():
    from src.common.seeds import seed_all

    seed_all(1337)
    a = torch.randn(4, 4)
    seed_all(1337)
    assert torch.equal(a, torch.randn(4, 4))


@pytest.mark.needs_torch
def test_no_float64_on_mps():
    """MPS does not support float64. Catch it here rather than mid-experiment."""
    if torch.backends.mps.is_available():
        with pytest.raises(TypeError):
            torch.zeros(2, dtype=torch.float64, device="mps")
