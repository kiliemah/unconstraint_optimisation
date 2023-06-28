import numpy as np


class qp(object):
    """
    Quadratic optimization
    """
    def __init__(self, X=np.array([0,0,0])) -> None:
        self.X = X

        # equality constraints
        self.A = np.array([1,1,1]).reshape(1,3)
        self.b = np.array([1])

        # inequality constraints
        self.inequality = np.array([self.f_1, self.f_2, self.f_3])
        # function to minimize
        self.func = self.f

    def f(self, X: np.ndarray):
        
        x = X[0]
        y = X[1]
        z = X[2]
        
        f = (x ** 2) + (y ** 2) + ((z+1) ** 2) 

        g = np.array([2*x,
                      2*y,
                      2*z + 1])
        
        H = np.array([[2,0,0],
                      [0,2,0],
                      [0,0,2]])
        
        return np.array([f,g,H], dtype=object)

    # Inequality constraints functions
    def f_1(self, X: np.ndarray):
        x = X[0]
        f = -x
        g = np.array([-1,0,0])

        H = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0]])
        
        return np.array([f,g,H], dtype=object)
    
    def f_2(self, X: np.ndarray):
        y = X[1]

        f = -y
        g = np.array([0,-1,0])

        H = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0]])
        
        return np.array([f,g,H], dtype=object)
    
    def f_3(self, X: np.ndarray):
        z = X[2]
        f = -z
        g = np.array([0,0,-1])

        H = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0]])
        
        return np.array([f,g,H], dtype=object)
        
    







class lp(object):

    """
    Linear programming
    """

    def __init__(self, X = np.array([0,0])) -> None:
        self.X = X

        # inequality constraints
        self.inequality = np.array([self.f_1, self.f_2, self.f_3, self.f_4])
        # function to minimize
        self.func = self.f


    def f(self, X: np.ndarray):
        
        x = X[0]
        y = X[1]

        f = - (x+y) # to minimize
        g = np.array([-1,
                      -1]) 
        H = np.array([[0,0],
                      [0,0]])
        
        return np.array([f,g,H], dtype=object)
        
    def f_1(self, X: np.ndarray):

        x = X[0]
        y = X[1]
        f = -x-y+1  
        g = np.array([-1,-1])
        H = np.array([[0,0],
                      [0,0]])
        
        return np.array([f,g,H], dtype=object)

    def f_2(self, X: np.ndarray):

        y = X[1]
        f = y-1
        g = np.array([0,1])
        H = np.array([[0,0],
                      [0,0]])
        
        return np.array([f,g,H], dtype=object)

    def f_3(self, X: np.ndarray):

        x = X[0]
        f = x-2
        g = np.array([1,0])
        H = np.array([[0,0],
                      [0,0]])
        
        return np.array([f,g,H], dtype=object)

    def f_4(self, X: np.ndarray):

        y = X[1]
        f = -y
        g = np.array([0,-1])
        H = np.array([[0,0],
                      [0,0]])
        
        return np.array([f,g,H], dtype=object)
    



