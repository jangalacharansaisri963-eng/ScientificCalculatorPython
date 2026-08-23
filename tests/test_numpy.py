"""
Unit tests and precision verification for pure-Python numpy module.
"""
import unittest
import ast
import os
import math

from functions import numpy as np

class TestPurePythonNumpy(unittest.TestCase):
    def test_zero_imports(self):
        """Verify that numpy.py contains zero import statements."""
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "functions", "numpy.py")
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 0, f"numpy.py must have zero imports, found: {imports}")

    def test_sqrt_precision(self):
        """Test square root precision across small, medium, and huge numbers."""
        test_values = [0.0, 1e-12, 1e-6, 0.25, 2.0, 3.0, 4.0, 100.0, 1234567.89, 1e16]
        for val in test_values:
            expected = math.sqrt(val)
            actual = np.sqrt(val)
            self.assertAlmostEqual(actual, expected, places=14, msg=f"Failed on sqrt({val})")

    def test_exp_and_log_precision(self):
        """Test exponential and natural logarithm precision and round-tripping."""
        test_values = [0.001, 0.1, 0.5, 1.0, 2.0, 3.14159, 10.0, 50.0, 200.0]
        for val in test_values:
            expected_exp = math.exp(val)
            actual_exp = np.exp(val)
            self.assertAlmostEqual(actual_exp, expected_exp, delta=expected_exp * 1e-14, msg=f"Failed on exp({val})")

        log_values = [1e-10, 1e-5, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1234567.0]
        for val in log_values:
            expected_log = math.log(val)
            actual_log = np.log(val)
            self.assertAlmostEqual(actual_log, expected_log, places=13, msg=f"Failed on log({val})")
            
            # Round-trip exp(log(x)) == x
            roundtrip = np.exp(np.log(val))
            self.assertAlmostEqual(roundtrip, val, delta=val * 1e-13)

    def test_trig_precision(self):
        """Test sine, cosine, and tangent precision."""
        angles = [-4.0 * math.pi, -math.pi, -math.pi/4, 0.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi, 100.0]
        for x in angles:
            self.assertAlmostEqual(np.sin(x), math.sin(x), places=14, msg=f"Failed sin({x})")
            self.assertAlmostEqual(np.cos(x), math.cos(x), places=14, msg=f"Failed cos({x})")
            if abs(math.cos(x)) > 1e-10:
                self.assertAlmostEqual(np.tan(x), math.tan(x), places=13, msg=f"Failed tan({x})")

    def test_statistical_stability_and_cancellation(self):
        """
        Test numerical stability of variance and std on shifted datasets
        which cause catastrophic cancellation in naive formulas.
        """
        # Ill-conditioned dataset: huge mean with small deviations
        base = 1_000_000_000.0
        data = [base + 1.0, base + 2.0, base + 3.0, base + 4.0, base + 5.0]
        
        # Exact mean is base + 3.0, exact population variance is 2.0
        arr = np.array(data)
        self.assertAlmostEqual(arr.mean(), base + 3.0, places=10)
        self.assertAlmostEqual(arr.var(ddof=0), 2.0, places=10)
        self.assertAlmostEqual(arr.std(ddof=0), math.sqrt(2.0), places=10)
        
        # Sample variance with ddof=1
        self.assertAlmostEqual(arr.var(ddof=1), 2.5, places=10)

    def test_linspace_exact_endpoints(self):
        """Verify linspace generates exact endpoints without floating point drift."""
        ls = np.linspace(0.0, 1.0, num=11)
        self.assertEqual(len(ls), 11)
        self.assertAlmostEqual(ls[0], 0.0)
        self.assertAlmostEqual(ls[-1], 1.0)
        self.assertAlmostEqual(ls[5], 0.5)

    def test_arange_precision(self):
        """Verify arange does not accumulate step drift."""
        ar = np.arange(0.0, 1.0, 0.1)
        self.assertEqual(len(ar), 10)
        self.assertAlmostEqual(ar[9], 0.9, places=14)
        
        # Negative step
        ar_neg = np.arange(5.0, 0.0, -1.0)
        self.assertEqual(ar_neg.tolist(), [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_ndarray_matrix_math(self):
        """Test array operations, broadcasting, and matrix dot products."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        
        # Addition
        c = a + b
        self.assertEqual(c.tolist(), [[6, 8], [10, 12]])
        
        # Scalar multiplication
        d = a * 2
        self.assertEqual(d.tolist(), [[2, 4], [6, 8]])
        
        # Matrix multiplication / dot product
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        m = a @ b
        self.assertEqual(m.tolist(), [[19, 22], [43, 50]])
        
        # Transpose
        self.assertEqual(a.T.tolist(), [[1, 3], [2, 4]])

    def test_zeros_ones_eye(self):
        """Test array initialization helpers."""
        z = np.zeros((2, 3))
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(z.tolist(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        o = np.ones((3,))
        self.assertEqual(o.tolist(), [1.0, 1.0, 1.0])
        
        i = np.eye(3)
        self.assertEqual(i.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

if __name__ == "__main__":
    unittest.main()
