import matplotlib.pyplot as plt
import numpy as np


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
        xx = np.linspace(-3, 3, 100)
        yy = np.linspace(-3, 3, 100)
        #zz = 

        X, Y = np.meshgrid(xx , yy)

        # computing Contour values
        i=0
        Z = []
        for y in yy:
            j=0
            axis_y=[]
            for x in xx:
                location = np.array([x,y])
                f_value=self.f(location)[0]
                axis_y.append(f_value)
                j+=1
            Z.append(axis_y)
            i+=1
        

        # Create a contour plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("contour lines of "+self.function_name)
        path_x = path_lists[:,0] 
        path_y = path_lists[:,1] 
        plt.plot(path_x, path_y, color="red", label="log-barrier")
        # Display the plot
        plt.legend()
        plt.show()    





    def path_lp(self, path_lists):
        """
        This function plot the path of a minimization algorthims on the contours of a given function
        Parameters
        ------------
        path_lists : np.ndarray 
            an array that contain the path array for each meathod
        """
        # Create x and y coordinate arrays
        xx = np.linspace(-1, 3, 100)
        yy = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(xx , yy)
        
        """
        # computing Contour values
        i=0
        Z = []
        for y in yy:
            j=0
            axis_y=[]
            for x in xx:
                location = np.array([x,y])
                f_value=self.f(location)[0]
                axis_y.append(f_value)
                j+=1
            Z.append(axis_y)
            i+=1
        
        # Create a contour plot
        plt.contourf(X, Y, Z, levels=20, cmap='viridis') #'viridis' 'coolwarm' 'RdGy' 'jet' 'inferno', 'plasma'
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("contour lines of "+self.function_name)
        path_x = path_lists[:,0] 
        path_y = path_lists[:,1]
        plt.plot(path_x, path_y, color="red", label=self.function_name)

        # Feasible region

        # ---- 1
        y1 = -xx + 1
        y2 = np.ones_like(xx)
        y3 = np.zeros_like(xx)
        x = np.full_like(yy, 2)

        
        plt.plot(xx, y1, color='blue', linewidth=2, linestyle='--', label='y ≥ -x + 1')
        plt.plot(xx, y2, color='blue', linewidth=2, linestyle='--', label='y ≤ 1')
        plt.plot(x, yy, color='blue', linewidth=2, linestyle='--', label='x ≤ 2')
        plt.plot(xx, y3, color='blue', linewidth=2, linestyle='--', label='y ≥ 0')
        """
        plt.imshow( ((Y >= 0) & (Y <= 1) & (X <= 2) & (Y>= -X + 1)) , 
                extent=(X.min(),X.max(),Y.min(),Y.max()), origin="lower", cmap="Greys", alpha = 0.9)

    

        # Display the plot
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
        plt.plot(iter, values_lists, color="blue", label="log - barrier")
        plt.title(" Function value by iteration - "+self.function_name)
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        plt.show()

        





