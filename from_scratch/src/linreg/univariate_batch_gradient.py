import numpy as np

class UnivariateBatchGradient:
    DEFAULT_ALPHA = 0.1
    INITIAL_THETA = np.array([0., 0.])

    def __init__(self, x, y):
        self.validate_vectors(x, y)
        self.X = self.feature_matrix(x)
        self.y = self.label_vector(y)
        self.data_size = len(x)
        self.computed_theta = None

    def validate_vectors(self, x, y):
        if (len(x) == 0):
            raise Exception('Cannot regress without data.')
        if (len(x) != len(y)):
            raise Exception('The vectors must be the same length.')

    def feature_matrix(self, x):
        x0 = np.ones(len(x))
        x1 = np.array(x)
        return np.vstack((x0, x1)).T

    def label_vector(self, y):
        return np.array(y).T

    def compute_cost_function(self, theta):
        sum = 0.0
        for i in range(self.data_size):
            h0 = self.X[i, :].dot(theta);
            sum = sum + (h0 - self.y[i])**2;
        return (sum / (2 * self.data_size));

    def gradient_descent(self, alpha=DEFAULT_ALPHA):
        theta = self.INITIAL_THETA
        while(True):
            new_theta = self.adjust_theta(theta.copy(), alpha)
            if (self.converges(theta, new_theta)):
                self.computed_theta = new_theta
                return new_theta
            else:
                theta = new_theta
    
    def adjust_theta(self, theta, alpha):
        for i in range(2):
            sum_of_differences = 0.0
            for j in range(self.data_size):
                difference = (self.evaluate_h(self.X[j,i], theta) - self.y[j]) * self.X[j,i]
                sum_of_differences += difference
            theta[i] = theta[i] - (alpha * (1/self.data_size) * sum_of_differences)
        return theta
    
    def evaluate_h(self, x, theta):
        return theta[0] + theta[1] * x
    
    def converges(self, theta, new_theta):
        old_cost = self.compute_cost_function(theta)
        new_cost = self.compute_cost_function(new_theta)
        return old_cost < new_cost

    def predict_label(self, x):
        if self.computed_theta is None:
            raise Execption('Gradient descent must be performed before predictions can be made.')
        else:
            return self.evaluate_h(x, self.computed_theta)