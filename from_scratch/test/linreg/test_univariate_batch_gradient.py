from src.linreg.univariate_batch_gradient import *
import pytest

class TestUnivariateBatchGradient:
    def test_arrays_validated_nonempty(self):
        x, y = [], []
        with pytest.raises(Exception):
            UnivariateBatchGradient(x, y)

    def test_arrays_validated_equal_size(self):
        x = [1, 2, 3]
        y = [1, 2]
        with pytest.raises(Exception):
            UnivariateBatchGradient(x, y)