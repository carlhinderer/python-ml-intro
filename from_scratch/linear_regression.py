from src.linreg.univariate_batch_gradient import *

def test_univariate_batch_gradient():
    x = [1, 2, 3, 4]
    y = [2, 3, 4, 5]
    g = UnivariateBatchGradient(x, y)

    theta = g.gradient_descent()
    prediction = g.predict_label(5)
    print('Prediction: ', prediction)

if __name__ == '__main__':
    test_univariate_batch_gradient()