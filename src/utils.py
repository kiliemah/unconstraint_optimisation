import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class plot(object):

    def __init__(self,f,function_name) -> None:
        self.f = f
        self.function_name = function_name
        
    
    def path_qp(self, path_lists):
        """
        This function plot the path of a minimization algorthims on the contours of a given function
        Parameters
        ------------
        path_lists : np.ndarray 
            an array that contain the path array for each meathod
        """

        # Create x and y coordinate arrays
        xx = np.linspace(0, 1, 100)
        yy = np.linspace(0, 1, 100)
        zz = np.linspace(0, 1, 100)

        X, Y ,Z= np.meshgrid(xx , yy, zz)
        
        # Feasible region
        
        # Define the constraints
        def constraint1(x, y, z):
            return x + y + z == 1

        def constraint2(x, y, z):
            return x >= 0 and y >= 0 and z >= 0


        # Check constraints for each point
        feasible_points = []
        for i in range(len(xx)):
            for j in range(len(yy)):
                for k in range(len(zz)):
                    if constraint1(xx[i], yy[j], zz[k]) and constraint2(xx[i], yy[j], zz[k]):
                        feasible_points.append((xx[i], yy[j], zz[k]))

        # Separate coordinates into x, y, z lists
        feasible_x = [p[0] for p in feasible_points]
        feasible_y = [p[1] for p in feasible_points]
        feasible_z = [p[2] for p in feasible_points]

        # Plot the feasible region
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(feasible_x, feasible_y, feasible_z, c='lightgreen', alpha=0.1)
       
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title("feasible region for "+self.function_name)
    
        path_x = path_lists[:,0] 
        path_y = path_lists[:,1]
        path_z = path_lists[:,2] 
        ax.plot(path_x, path_y, path_z, color="red", linewidth=2.5, label="interior point path", alpha=1)
        
        # Display the plot
        plt.legend()
        plt.show()    



    def path_lp(self, path_lists):
        # Create x and y coordinate arrays
        xx = np.linspace(-1, 3, 300)
        yy = np.linspace(-2, 2, 300)
        X, Y = np.meshgrid(xx , yy)
    
        
        # Feasible region
        
        # Define the constraints
        def constraint1(x, y):
            return y >= -x +1

        def constraint2(x, y):
            return y <= 1
        
        def constraint3(x, y):
            return x <= 2
        
        def constraint4(x, y):
            return y >= 0


        # Check constraints for each point
        feasible_points = []
        for i in range(len(xx)):
            for j in range(len(yy)):
                    if constraint1(xx[i], yy[j]) and constraint2(xx[i], yy[j]) and constraint3(xx[i], yy[j]) and constraint4(xx[i], yy[j]):
                        feasible_points.append((xx[i], yy[j]))

        # Separate coordinates into x, y, z lists
        feasible_x = [p[0] for p in feasible_points]
        feasible_y = [p[1] for p in feasible_points]
    

        # Plot the feasible region
        fig, ax = plt.subplots()
        ax.scatter(feasible_x, feasible_y, c='lightgreen', alpha=0.3)

        path_x = path_lists[:,0] 
        path_y = path_lists[:,1]
        ax.plot(path_x, path_y, color="red", linewidth=2.5, label="interior point path", alpha=1)

        y1 = -xx + 1
        y2 = np.ones_like(xx)
        y3 = np.zeros_like(xx)
        x = np.full_like(yy, 2)
        
        plt.plot(xx, y1, color='blue', linewidth=1, linestyle='--', label='y ≥ -x + 1')
        plt.plot(xx, y2, color='purple', linewidth=1, linestyle='--', label='y ≤ 1')
        plt.plot(x, yy, color='pink', linewidth=1, linestyle='--', label='x ≤ 2')
        plt.plot(xx, y3, color='orange', linewidth=1, linestyle='--', label='y ≥ 0')

        


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
    
        plt.show()


       
    
    def iteration(self, values_lists):
        """
        This function plot the function value at each iteration
        ------------
        values_lists : np.ndarray
            an array that contain the minimization values array for each meathod
        """

        iter = range(values_lists.shape[0])
        plt.plot(iter, values_lists, color="blue", label="f(x)")
        plt.title(" Function value by iteration - "+self.function_name)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.show()

        





