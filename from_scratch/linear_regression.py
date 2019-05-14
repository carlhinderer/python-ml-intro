from src.linreg.univariate_batch_gradient import *

def test_univariate_batch_gradient():
    x = [1, 2, 3, 4]
    y = [2, 3, 4, 5]
    g = UnivariateBatchGradient(x, y)

    theta = np.array([[1.], [1.]])
    cost = g.compute_cost_function(theta)
    print(cost)

if __name__ == '__main__':
    test_univariate_batch_gradient()