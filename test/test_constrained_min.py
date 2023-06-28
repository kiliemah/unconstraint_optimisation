import sys
import os
import numpy as np

# Get the src directory
subfolder_path = os.path.dirname("src")
sys.path.append(subfolder_path)

from src import constrained_min, utils
from example import *
import unittest


def test_qp():
    x_0 = np.array([0.1, 0.2, 0.7])
    
    qp_problem = qp()
    func = qp_problem.func
    ineq_constraints = qp_problem.inequality
    eq_constraints_mat = qp_problem.A
    eq_constraints_rhs = qp_problem.b


    opt = constrained_min.interior_pt(func = func,
                                      ineq_constraints = ineq_constraints,
                                      eq_constraints_mat = eq_constraints_mat,
                                      eq_constraints_rhs = eq_constraints_rhs,
                                      x_0 = x_0)
    opt.minimize()
    path = opt.x_path
    function_value = opt.f_path_loc
    plot = utils.plot(func, "Linear Programming")
    #plot.path_lp(path)
    plot.iteration(function_value)


def test_lp():
    x_0 = np.array([0.5, 0.75])
    
    lp_problem = lp()
    func = lp_problem.func
    ineq_constraints = lp_problem.inequality

    opt = constrained_min.interior_pt(func = func,
                                      ineq_constraints = ineq_constraints, 
                                      x_0 = x_0)
    opt.minimize()
    path = opt.x_path
    function_value = opt.f_path_loc
    
    #plot = utils.plot(func, "Quadratic Programming")
    #plot.path_lp(path)
    #plot.iteration(function_value)






#test_qp()


test_lp()