import numpy as np

class interior_pt(object):

    """
    parameter :
    ---------- 

        - func : python function to minimize
        - ineq_constraints : the list of inequality constraints specified by the Python list of functions
        - eq_constraints_mat : Matrix A of the equality constraint Ax = b
        - eq_constraints_rhs : vector b of the equality constraint Ax = b

        
    returned values : 
    ----------------
        -  x_path : nd.array of shape (n,k) where n is the number of explored location 
                    and k is the dimenssion of x_0
                    Location x , and function value f(x) at each iteration during the minimization process

        - f_path_loc : ndarray of shape (n,)
                    The value of f at a given location in the optimization path

        - final_x : nd.array of the size of x_0
                    Final location x that minimize the function

        - final_f_x : float
                    The function value at the final location that minimize f (the min of the function)

        - status : bool
                    Boolean (True or False) that say if the optimization process is a success or failure
    """

    def __init__(self, 
                 func,
                 ineq_constraints = 0, 
                 eq_constraints_mat = 0, 
                 eq_constraints_rhs = 0, 
                 x_0 = 0) -> None:
        
        # Initialization attributes
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.x_0 = x_0

        # stopping log barrier
        self.eps = 10**(-8)

        # Newton parameters
        self.obj_tol = 10**(-8)
        self.param_tol = 10**(-12)
        self.max_iter = 100 
        self.c_1 = 0.01
        self.back = 0.5
        
        # results attributes
        self.x_path = None
        self.f_path_loc = None
        self.final_x = None
        self.final_f_x = None
        self.inquality_value  = None
        self.equality_value  = None
        

    
    def phi(self, x : np.ndarray):
        """
        Defining function phi
        """
        
        sum_value = 0
        sum_gradient = 0
        sum_H = 0

        for function in self.ineq_constraints:
            func = function(x)
            
            # computing phi(x)
            try:
                sum_value += np.log(-func[0])
            except:
                sum_value = None

            # computing gradient of phi(x)
            a = -(1/func[0])
            sum_gradient += a * func[1]

            # computing Hessian of phi
            b = ((1/func[0])**2) * np.dot(func[1],func[1])
            sum_H += b + a * func[2]


        f = - sum_value
        g = sum_gradient
        H = sum_H
        return np.array([f,g,H], dtype=object)


    
    def log_barrier(self, t, func, phi, x):
        """
        Defining log-barrier function
        """

        func_ = func(x)
        phi_ = phi(x)

        f = t * func_[0] + phi_[0]
        g = t * func_[1] + phi_[1]
        H = t * func_[2] + phi_[2]

        return np.array([f,g,H], dtype=object)
    
    
    def Newton_constraint(self, t, f, phi, x_0, obj_tol, param_tol, max_iter, c_1, back, A, b):
        

        def find_step(x, f_i, g_i, p):
            """
            function that compute the next location according the wolf condition

            Paratmeters
            -----------
            x : current location
            f_i : f at the current location x
            p : founded direction
            g_i :  gradient at the current location i
            alpha_space : np.ndarray where values are between 0 and 1
            
            return 
            -----------
            x_step : next location to investigate
            """
            
            alpha = 1
            while True:
                
                x_step = x + alpha * p
                f_x_step = self.log_barrier(t,f,phi,x_step)[0]
                
                wolf_cond_1 = f_i + (c_1 * alpha * np.dot(g_i, p))
                if f_x_step <=  wolf_cond_1 :
                    return x_step, f_x_step
                alpha = back * alpha


        status = False
        stop_check = 1000 # init objectif tolerence stopping condition
        distance_check = 1000 # init distance tolerence stopping condition 
        x = [x_0] # list locations visited
        f_x_0 = self.log_barrier(t,f,phi,x_0)[0]
        f_x = [f_x_0] # list of function value at the location isited 
        p = None # direction to visit
        print(" ----- Newton Method ----- ")
        for i in range(max_iter):
            if  distance_check < param_tol or np.abs(stop_check) < obj_tol: # Checking distance between 2 last locations
                # Up the status flag in case we reached a minimum by distance stopping condition
                status = True
                break
            else :
                
                zero = np.zeros((A.shape[0],A.shape[0]))
                g, H = self.log_barrier(t,f,phi,x[-1])[1:]

                M = np.block([[H,  A.T],
                              [A, zero]])
                
                
                G = np.block([g, np.dot(A,x[-1]) - b])
        
                try:
                    p_w = np.linalg.solve(M, -G)
                except:
                    break

                p = p_w[:A.T.shape[0]]
                
                # Newton decrement stopping condition
                newton_dcr = f_x[-1] - f_x[-1] + (1/2) * np.dot(np.dot(p,H),p)
                if newton_dcr < self.obj_tol:
                    break

                next_x, f_next_x = find_step(x[-1], f_x[-1], g, p)
                x.append(next_x)
                f_x.append(f_next_x)
                stop_check = np.abs(f_x[-1] - f_x[-2])
                distance_check = np.linalg.norm(x[-1] - x[-2])

        final_x = x[-1]
        final_f_x = f(x[-1])[0]  # value of the true objectiv function
        x_path = np.array(x)
        f_path_loc = np.array(f_x) # Inner path of newton method for log barrier function

        return np.array([final_x, final_f_x, x_path, f_path_loc, status])
    

    def Newton_unconstraint(self, t,f,phi, x_0, obj_tol, param_tol, max_iter, c_1, back, A, b):
        

        def find_step(x, f_i, g_i, p):
            """
            function that compute the next location according the wolf condition

            Paratmeters
            -----------
            x : current location
            f_i : f at the current location x
            p : found direction
            g_i :  gradient at the current location i
            alpha_space : np.ndarray where values are between 0 and 1
            
            return 
            -----------
            x_step : next location to investigate
            """
            alpha = 1
            while True:
                x_step = x + alpha * p
                f_x_step = self.log_barrier(t,f,phi,x_step)[0]
                wolf_cond_1 = f_i + (c_1 * alpha * np.dot(g_i, p))
                if f_x_step <=  wolf_cond_1 :
                    return x_step, f_x_step
                alpha = back * alpha


        status = False
        stop_check = 1000 # init objectif tolerence stopping condition
        distance_check = 1000 # init distance tolerence stopping condition 
        x = [x_0] # list locations visited
        f_x_0 = self.log_barrier(t,f,phi,x_0)[0]
        f_x = [f_x_0] # list of function value at the location isited 
        p = None # direction to visit
        for i in range(max_iter):

            if  distance_check < param_tol or np.abs(stop_check) < obj_tol: # Checking distance between 2 last locations
                # Up the status flag in case we reached a minimum by distance stopping condition
                status = True
                break
            else :
                g, H = self.log_barrier(t,f,phi,x[-1])[1:]
                print(H)
                try:
                    p = np.linalg.solve(H, -g)
                except:
                    break
                    
                    
                # Newton decrement stopping condition
                newton_dcr = (-1/2) * np.dot(np.dot(p,H),p)
                if newton_dcr < self.obj_tol:
                    break

                next_x, f_next_x = find_step(x[-1], f_x[-1], g, p)
                print(next_x)
                x.append(next_x)
                f_x.append(f_next_x)
                stop_check = np.abs(f_x[-1] - f_x[-2])
                distance_check = np.linalg.norm(x[-1] - x[-2])

        final_x = x[-1]
        final_f_x = f(x[-1])[0]  # value of the true objectiv function
        x_path = np.array(x)
        f_path_loc = np.array(f_x) # Inner path of newton method for log barrier function
        return np.array([final_x, final_f_x, x_path, f_path_loc, status], dtype=object)
    
            
    
    def GD(self, t,f,phi, x_0, obj_tol, param_tol, max_iter, c_1, back, A, b):
        """
        This function compute the minimimun ot the function according gradient descent method
        It update the corresponding attribute 

            self.path 
            self.final_x 
            self.final_f_x 
            self.status 

        which are the return values of the optimization process

        """
        def find_step(x, f_i, g_i, p):
            """
            function that compute the next location according the wolf condition

            Paratmeters
            -----------
            x : current location
            f_i : f at the current location x
            p : found direction
            g_i :  gradient at the current location i
            alpha_space : np.ndarray where values are between 0 and 1
            
            return 
            -----------
            x_step : next location to investigate
            """
            alpha = 1
            while True:
                x_step = x + alpha * p
                f_x_step = self.log_barrier(t,f,phi,x_step)[0]
                wolf_cond_1 = f_i + (c_1 * alpha * np.dot(g_i, p))
                if f_x_step <=  wolf_cond_1 :
                    return x_step, f_x_step
                alpha = back * alpha

        status = False
        stop_check = 1000 # init objectif tolerence stopping condition
        distance_check = 1000 # init distance tolerence stopping condition 
        x = [x_0] # list locations visited
        f_x_0 = self.log_barrier(t,f,phi,x_0)[0]
        f_x = [f_x_0] # list of function value at the location isited 
        p = None # direction to visit
        for i in range(self.max_iter):

            if stop_check < self.obj_tol or distance_check < self.param_tol:
                # Up the status flag in case we reached a minimum by stopping condition
                self.status = True
                break
            else :
                g= self.log_barrier(t,f,phi,x[-1])[1]
                p = -g
                next_x, f_next_x = find_step(x[-1],f_x[-1], g, p)
                x.append(next_x)
                f_x.append(f_next_x)
                stop_check = np.abs(f_x[-1] - f_x[-2])
                distance_check = np.linalg.norm(x[-1] - x[-2])
        
        # In case the number of iteration is reached OR we break the loop by a stop condtition
        # We update the finals results
        final_x = x[-1]
        final_f_x = f(x[-1])[0]  # value of the true objectiv function
        x_path = np.array(x)
        f_path_loc = np.array(f_x) # Inner path of newton method for log barrier function
        return np.array([final_x, final_f_x, x_path, f_path_loc, status], dtype=object)
    
        

    def minimize(self):

        print("************ Log-barrier Algorithm **********")
        t = 1
        m = self.ineq_constraints.shape[0]
        A = self.eq_constraints_mat
        b = self.eq_constraints_rhs
        x = [self.x_0]
        f_x = [self.func(self.x_0)[0]]
        i=0
        mu = 10
        #while t==1:
        while (m/t) >= self.eps:

            print("> Iteration ",i,":")
            print("     - location :",x[-1])
            print("     - f(x)     :",f_x[-1])

            
            if isinstance(self.eq_constraints_mat, np.ndarray):

                newton = self.Newton_constraint(t,
                                    self.func,
                                    self.phi,
                                    x[-1],
                                    self.obj_tol,
                                    self.param_tol,
                                    self.max_iter,
                                    self.c_1,
                                    self.back,
                                    A,
                                    b)
            else :
        
                newton = self.GD(t,
                                    self.func,
                                    self.phi,
                                    x[-1],
                                    self.obj_tol,
                                    self.param_tol,
                                    self.max_iter,
                                    self.c_1,
                                    self.back,
                                    A,
                                    b)
            
            
            x.append(newton[0])
            f_x.append(newton[1])

            i += 1
            t = mu * t

        ineq = []
        for const in self.ineq_constraints:
            ineq.append(const(x[-1])[0])
        self.inquality_value = np.array(ineq)

        if  isinstance(self.eq_constraints_mat, np.ndarray):
            self.equality_value = np.dot(A,x[-1])

        self.final_x = x[-1]
        self.final_f_x = f_x[-1]
        self.x_path = np.array(x)
        self.f_path_loc = np.array(f_x)

        print("> Iteration ",i,":")
        print("     - location :",x[-1])
        print("     - f(x)     :",f_x[-1])
        print("     - equality constraint     :",self.equality_value)
        print("     - inequality constraint     :",self.inquality_value)
        
