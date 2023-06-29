import sys
import os
import numpy as np

# Get the src directory
subfolder_path = os.path.dirname("src")
sys.path.append(subfolder_path)

from src import constrained_min, utils
from example import *
import unittest



class Test(unittest.TestCase):
    """
    This class compute the minimization for 2 constraint problem:
    """

    def test_qp(self):

        x_0 = np.array([0.1, 0.2, 0.7])
        
        # Problem instance for example function qp
        qp_problem = qp()
        func = qp_problem.func
        ineq_constraints = qp_problem.inequality
        eq_constraints_mat = qp_problem.A
        eq_constraints_rhs = qp_problem.b

        # Optimization
        opt = constrained_min.interior_pt(func = func,
                                        ineq_constraints = ineq_constraints,
                                        eq_constraints_mat = eq_constraints_mat,
                                        eq_constraints_rhs = eq_constraints_rhs,
                                        x_0 = x_0)
        opt.minimize()
        path = opt.x_path
        function_value = opt.f_path_loc

        #Plotting
        plot = utils.plot(func, "Quadratic Programming")
        plot.path_qp(path)
        plot.iteration(function_value)


    def test_lp(self):

        x_0 = np.array([0.5, 0.75])
        
        # Problemn instance from example function lp()
        lp_problem = lp()
        func = lp_problem.func
        ineq_constraints = lp_problem.inequality

        # Optimization
        opt = constrained_min.interior_pt(func = func,
                                        ineq_constraints = ineq_constraints, 
                                        x_0 = x_0)
        opt.minimize()
        path = opt.x_path
        function_value = opt.f_path_loc
        
        # Plotting
        plot = utils.plot(func, "Linear Programming")
        plot.path_lp(path)
        plot.iteration(function_value)






# Testing
test = Test()

test.test_qp()
test.test_lp()