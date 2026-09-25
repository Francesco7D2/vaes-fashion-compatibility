import pytest

from src.utils.device_utilities import choose_device


@pytest.mark.parametrize(
    ("preference", "gpu_count", "expected"),
    [
        ("auto", 0, "cpu"),
        ("auto", 1, "gpu"),
        ("cpu", 2, "cpu"),
        ("gpu", 1, "gpu"),
        ("AUTO", 1, "gpu"),
    ],
)
def test_choose_device(preference, gpu_count, expected):
    assert choose_device(preference, gpu_count) == expected


def test_choose_device_rejects_missing_or_invalid_gpu():
    with pytest.raises(RuntimeError, match="found none"):
        choose_device("gpu", 0)
    with pytest.raises(ValueError, match="auto, cpu, gpu"):
        choose_device("tpu", 1)
