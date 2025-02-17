import unittest
import numpy as np
from src.main.assignment_2 import (
    neville, newton_forward_table, newton_forward_polynomial, hermite_interpolation, cubic_spline
)

class TestAssignment2(unittest.TestCase):
    
    def test_neville(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        x_target = 3.7
        result = neville(x_vals, y_vals, x_target)
        self.assertAlmostEqual(result, 1.554, places=3)
    
    def test_newton_forward_polynomial(self):
        x_values = np.array([7.2, 7.4, 7.5, 7.6])
        y_values = np.array([23.5492, 25.3913, 26.8224, 27.4589])
        diff_table = newton_forward_table(x_values, y_values)
        x_target = 7.3
        approximation = newton_forward_polynomial(x_values, diff_table, 3, x_target)
        self.assertAlmostEqual(approximation, 24.474, places=3)
    
    def test_hermite_interpolation(self):
        x_vals = np.array([3.6, 3.8, 3.9])
        f_vals = np.array([1.675, 1.436, 1.318])
        f_prime_vals = np.array([-1.195, -1.188, -1.182])
        H = hermite_interpolation(x_vals, f_vals, f_prime_vals)
        self.assertEqual(H.shape, (6, 6))
    
    def test_cubic_spline(self):
        x_vals = np.array([2, 5, 8, 10], dtype=float)
        f_vals = np.array([3, 5, 7, 9], dtype=float)
        A, b, x = cubic_spline(x_vals, f_vals)
        self.assertEqual(A.shape, (4, 4))
        self.assertEqual(len(b), 4)
        self.assertEqual(len(x), 4)

if __name__ == "__main__":
    unittest.main()
