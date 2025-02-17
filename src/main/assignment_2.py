import numpy as np

def neville(x_vals, y_vals, x):
    n = len(x_vals)
    P = [[0] * n for _ in range(n)]
    
    for i in range(n):
        P[i][i] = y_vals[i]
    
    for j in range(1, n):
        for i in range(n - j):
            P[i][i + j] = ((x - x_vals[i + j]) * P[i][i + j - 1] - 
                            (x - x_vals[i]) * P[i + 1][i + j]) / (x_vals[i] - x_vals[i + j])
    
    return P[0][n - 1]  

def newton_forward_table(x_values, y_values):
    n = len(x_values)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_values  
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1])
    
    return diff_table

def newton_forward_polynomial(x_values, diff_table, degree, x_target):
    h = x_values[1] - x_values[0]
    u = (x_target - x_values[0]) / h

    result = diff_table[0, 0]
    term = 1
    for i in range(1, degree + 1):
        term *= (u - (i - 1)) / i
        result += term * diff_table[0, i]

    return result

def hermite_interpolation(x_vals, f_vals, f_prime_vals):
    n = len(x_vals)
    H = np.zeros((2 * n, 2 * n))

    for i in range(n):
        H[2 * i][0] = x_vals[i]
        H[2 * i + 1][0] = x_vals[i]
        H[2 * i][1] = f_vals[i]
        H[2 * i + 1][1] = f_vals[i]
        H[2 * i + 1][2] = f_prime_vals[i]

    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            if H[i + j][0] != H[i][0]:
                H[i][j] = (H[i + 1][j - 1] - H[i][j - 1]) / (H[i + j][0] - H[i][0])
    
    return H

def cubic_spline(x_vals, f_vals):
    n = len(x_vals) - 1
    h = np.diff(x_vals)

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    A[0, 0] = 1  
    A[n, n] = 1  

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]  
        A[i, i] = 2 * (h[i - 1] + h[i])  
        A[i, i + 1] = h[i]  
        b[i] = (3 / h[i]) * (f_vals[i + 1] - f_vals[i]) - (3 / h[i - 1]) * (f_vals[i] - f_vals[i - 1])

    x = np.linalg.solve(A, b)
    
    return A, b, x

