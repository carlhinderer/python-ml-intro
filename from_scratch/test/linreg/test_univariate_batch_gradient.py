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

    def test_cost_function(self):
        x = [1, 2, 3, 4]
        y = [2, 3, 4, 5]
        g = UnivariateBatchGradient(x, y)

        theta = np.array([[1.], [1.]])
        cost = g.compute_cost_function(theta)
        assert cost == 0

        theta = np.array([[0.], [0.]])
        cost = g.compute_cost_function(theta)
        assert cost == 6.75

    def test_gradient_descent(self):
        x = [1, 2, 3, 4]
        y = [2, 3, 4, 5]
        g = UnivariateBatchGradient(x, y)

        theta = g.gradient_descent()
        cost = g.compute_cost_function(theta)
        assert cost <= 0.01

    def test_theta_computed_before_prediction(self):
        x = [1, 2, 3, 4]
        y = [2, 3, 4, 5]
        g = UnivariateBatchGradient(x, y)

        with pytest.raises(Exception):
            g.predict(5)

    def test_prediction(self):
        x = [1, 2, 3, 4]
        y = [2, 3, 4, 5]
        g = UnivariateBatchGradient(x, y)
        g.gradient_descent()

        prediction = g.predict_label(5)
        assert abs(6 - prediction) <= 0.05