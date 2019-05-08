from src.linreg.univariate_batch_gradient import *

def test_univariate_batch_gradient():
	g = UnivariateBatchGradient()
	g.print_gradients()

if __name__ == '__main__':
	test_univariate_batch_gradient()