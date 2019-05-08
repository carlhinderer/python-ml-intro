import numpy as np

class UnivariateBatchGradient:
    XS = [1, 2, 3, 4]
    YS = [2, 3, 4, 5]

    ALPHA = 0.1
    INITIAL_THETA = np.array([0, 0])
    CONVERGENCE_THRESHOLD = 0.0001

    def __init__(self, x, y):
        self.validate_vectors(x, y)
        self.x = np.array(x)
        self.y = np.array(y)

    def validate_vectors(self, x, y):
        if (len(x) == 0):
            raise Exception('Cannot regress without data.')
        if (len(x) != len(y)):
            raise Exception('The vectors must be the same length.')

    def print_gradients(self):
        t0, t1 = 0, 0
        iteration = 0
        while(True):
            print('iteration: %s, t0: %s, t1: %s' % (iteration, t0, t1))
            new_t0 = self.adjust_t0(t0, t1)
            new_t1 = self.adjust_t1(t0, t1)
            if (self.converges(t0, new_t0) and self.converges(t1, new_t1)):
                break
            else:
                iteration += 1
                t0, t1 = new_t0, new_t1
    
    def adjust_t0(self, t0, t1):
        sum_of_differences = 0.0
        for i in range(len(self.XS)):
            difference = self.evaluate_h0(self.XS[i], t0, t1) - self.YS[i]
            sum_of_differences += difference
        return t0 - (self.ALPHA * (1/len(self.XS)) * sum_of_differences)
    
    def adjust_t1(self, t0, t1):
        sum_of_differences = 0.0
        for i in range(len(self.XS)):
            difference = (self.evaluate_h0(self.XS[i], t0, t1) - self.YS[i]) * self.XS[i]
            sum_of_differences += difference
        return t1 - (self.ALPHA * (1/len(self.XS)) * sum_of_differences)
    
    def evaluate_h0(self, x, t0, t1):
        return t0 + t1 * x
    
    def converges(self, t, new_t):
        return abs(t - new_t) <= self.CONVERGENCE_THRESHOLD