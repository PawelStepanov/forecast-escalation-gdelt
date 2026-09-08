import math
from src.metrics import rmse, smape, mase


def test_rmse_zero():
    assert rmse([1, 2], [1, 2]) == 0.0


def test_smape_zero():
    assert smape([0, 2], [0, 2]) == 0.0


def test_mase():
    assert math.isclose(mase([2, 4], [1, 5], scale=2.0), 0.5)
