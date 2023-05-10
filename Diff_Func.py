import numpy as np
import numba
from numba import njit
from numba import njit, prange
#help from CHATGPT and 
#https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lid_driven_cavity_python_simple.py
#Function of the laplace operator of a grid
@numba.jit(nopython=True, parallel=True)
def laplace_operator_dx(grid,dx):
    diff = np.zeros_like(grid)
    diff[1:-1, 1:-1] = (grid[1:-1, 0:-2]+grid[0:-2, 1:-1]-4*grid[1:-1, 1:-1]+grid[1:-1, 2:]+grid[2:  , 1:-1]) / (dx**2)
    return diff

@numba.jit(nopython=True, cache=True)
def laplace_operator_dy(grid,dy):
    diff = np.zeros_like(grid)
    diff[1:-1, 1:-1] = (grid[1:-1, 0:-2]+grid[0:-2, 1:-1]-4*grid[1:-1, 1:-1]+grid[1:-1, 2:]+grid[2:  , 1:-1]) / (dy**2)
    return diff

#central difference in x direction of a grid
@numba.jit(nopython=True, cache=True)
def central_difference_u(grid, dx):
    diff = np.zeros_like(grid)
    diff[1:-1, 1:-1] = (grid[1:-1, 2: ]-grid[1:-1, 0:-2]) / (2 * dx)
    return diff

#central difference in y direction of a grid
@numba.jit(nopython=True, cache=True)
def central_difference_v(grid, dy):
    diff = np.zeros_like(grid)
    diff[1:-1, 1:-1] = (grid[2:  , 1:-1]-grid[0:-2, 1:-1]) / (2 * dy)
    return diff

@numba.jit(nopython=True, cache=True)
def finite_difference_u(grid, dx):
    diff = np.zeros_like(grid)
    diff[1:-1, 1:-1] = (grid[1:-1, 2:] - grid[1:-1, :-2]) / (2 * dx)
    return diff

@numba.jit(nopython=True, cache=True)
def finite_difference_v(grid, dy):
    diff = np.zeros_like(grid)
    diff[1:-1, 1:-1] = (grid[2:, 1:-1] - grid[:-2, 1:-1]) / (2 * dy)
    return diff

#Function of a Gradient of a grid
@numba.jit(nopython=True, cache=True)
def gradient(grid):
    grad_y, grad_x = np.gradient(grid)
    return grad_x, grad_y

@numba.jit(nopython=True, cache=True)
def solve_pressure(p, rhs, dx, max_iter=1000, omega=1.7, tol=1e-7):
    nx, ny = p.shape
    dx2 = dx ** 2
    p_temp = np.copy(p)  # Create a temporary array for storing updated values
    for k in range(max_iter):
        max_diff = 0.0  # Initialize the maximum difference (so numba can compile the code -> np.max() is not supported
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                p_old = p[i, j]
                p_new = (1 - omega) * p_old + omega * 0.25 * (p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1] - dx2 * rhs[i, j])
                p_temp[i, j] = p_new  # Assign updated values to the temporary array
                diff = np.abs(p_new - p_old)  # Calculate the difference
            
                if diff > max_diff:
                    max_diff = diff
                p = neumann_bc(p_temp)
        if max_diff < tol:
            break
    return p

@numba.jit(nopython=True, cache=True)
def neumann_bc(array):
    # Apply Neumann boundary condition
    array[:, -1] = array[:, -2] #(right boundary condition)
    array[-1, :] = array[-2, :] #(bottom boundary condition)
    array[:, 0] = array[:, 1] #(left boundary condition)
    #array[0, :] = array[1, :] #(top boundary condition)
    return array