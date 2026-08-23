"""
Comprehensive unit tests for mathematical sub-functions and physics models.
"""
import unittest
import math
import cmath

from functions import basic_math
from functions import trig
from functions import hyperbolic
from functions import logarithms
from functions import roots
from functions import factorial
from functions import factors
from functions import divisibility
from functions import integers
from functions import binomial
from functions import statistics
from functions import dimensions
from functions import rationals
from functions import simplify
from functions import complex_numbers
from functions import compare
from functions import derivative
from functions import custom_math
from functions import physics

class TestMathematicalFunctions(unittest.TestCase):
    # Basic math
    def test_basic_math(self):
        self.assertEqual(basic_math.add(10, 5), 15)
        self.assertEqual(basic_math.subtract(10, 5), 5)
        self.assertEqual(basic_math.multiply(10, 5), 50)
        self.assertEqual(basic_math.divide(10, 5), 2.0)
        self.assertEqual(basic_math.power(2, 3), 8)
        self.assertEqual(basic_math.modulo(10, 3), 1)
        self.assertEqual(basic_math.sign_val(-42), -1)
        self.assertEqual(basic_math.sign_val(42), 1)
        self.assertEqual(basic_math.sign_val(0), 0)

    # Trigonometry
    def test_trigonometry(self):
        self.assertAlmostEqual(trig.sin_func(90, mode="DEG"), 1.0)
        self.assertAlmostEqual(trig.cos_func(0, mode="DEG"), 1.0)
        self.assertAlmostEqual(trig.tan_func(45, mode="DEG"), 1.0)
        self.assertAlmostEqual(trig.asin_func(1, mode="DEG"), 90.0)
        self.assertAlmostEqual(trig.acos_func(1, mode="DEG"), 0.0)
        self.assertAlmostEqual(trig.atan_func(1, mode="DEG"), 45.0)

    # Hyperbolic
    def test_hyperbolic(self):
        self.assertAlmostEqual(hyperbolic.sinh_func(0), 0.0)
        self.assertAlmostEqual(hyperbolic.cosh_func(0), 1.0)
        self.assertAlmostEqual(hyperbolic.tanh_func(0), 0.0)
        self.assertAlmostEqual(hyperbolic.asinh_func(0), 0.0)
        self.assertAlmostEqual(hyperbolic.acosh_func(1), 0.0)

    # Logarithms & Exponential
    def test_logarithms(self):
        self.assertAlmostEqual(logarithms.ln_func(math.e), 1.0)
        self.assertAlmostEqual(logarithms.log10_func(1000), 3.0)
        self.assertAlmostEqual(logarithms.log2_func(32), 5.0)
        self.assertAlmostEqual(logarithms.log_base_func(81, base=3), 4.0)
        self.assertAlmostEqual(logarithms.exp_func(1), math.e)

    # Roots
    def test_roots(self):
        self.assertAlmostEqual(roots.sqrt_func(144), 12.0)
        self.assertAlmostEqual(roots.cbrt_func(27), 3.0)
        self.assertAlmostEqual(roots.nth_root_func(1024, 10), 2.0)
        self.assertTrue(roots.is_perfect_square(64))
        self.assertFalse(roots.is_perfect_square(65))

    # Factorials, Combinatorics & Binomial Theorem
    def test_combinatorics(self):
        self.assertEqual(factorial.factorial_func(6), 720)
        self.assertEqual(factorial.double_factorial(5), 15)  # 5 * 3 * 1
        self.assertEqual(factorial.subfactorial(4), 9)
        self.assertEqual(binomial.nCr(10, 3), 120)
        self.assertEqual(binomial.nPr(6, 2), 30)
        self.assertEqual(binomial.catalan_number(4), 14)
        
        # Binomial Theorem expansions: (2 + 3)^4 = 5^4 = 625
        exp = binomial.binomial_expansion(4, 2, 3)
        self.assertAlmostEqual(exp["sum"], 625.0)
        self.assertEqual(exp["coefficients"], [1, 4, 6, 4, 1])
        
        # Symbolic Binomial expansion string
        str_exp = binomial.binomial_expansion_str(3, "x", "y")
        self.assertEqual(str_exp, "x^3 + 3*x^2*y + 3*x*y^2 + y^3")
        
        # Binomial probability distributions
        # P(X=2) for n=4, p=0.5: 6 * (0.5)^4 = 0.375
        self.assertAlmostEqual(binomial.binomial_pmf(4, 2, 0.5), 0.375)
        # P(X<=2) for n=4, p=0.5: (1 + 4 + 6)/16 = 11/16 = 0.6875
        self.assertAlmostEqual(binomial.binomial_cdf(4, 2, 0.5), 0.6875)
        
        stats = binomial.binomial_stats(100, 0.25)
        self.assertAlmostEqual(stats["mean"], 25.0)
        self.assertAlmostEqual(stats["variance"], 18.75)

    # Number theory & Primes
    def test_number_theory(self):
        self.assertTrue(factors.is_prime(97))
        self.assertFalse(factors.is_prime(91))
        self.assertEqual(factors.prime_factors(360), [2, 2, 2, 3, 3, 5])
        self.assertEqual(divisibility.gcd_func(48, 18), 6)
        self.assertEqual(divisibility.lcm_func(12, 15), 60)
        self.assertEqual(divisibility.mod_inverse(3, 11), 4)

    # Integers
    def test_integers(self):
        self.assertTrue(integers.is_even(42))
        self.assertTrue(integers.is_odd(43))
        self.assertEqual(integers.sum_digits(12345), 15)
        self.assertEqual(integers.reverse_integer(1234), 4321)
        self.assertTrue(integers.is_palindrome_number(12321))
        self.assertEqual(integers.digital_root(9875), 2)  # 9+8+7+5=29->11->2

    # Rationals & Radicals
    def test_rationals_and_radicals(self):
        self.assertEqual(rationals.simplify_fraction(18, 24), (3, 4))
        self.assertEqual(simplify.simplify_radical(18), (3, 2))  # 3 * sqrt(2)
        self.assertEqual(simplify.format_simplified_radical(18), "3*sqrt(2)")

    # Complex Numbers
    def test_complex_numbers(self):
        z = complex_numbers.make_complex(3, 4)
        self.assertAlmostEqual(complex_numbers.complex_magnitude(z), 5.0)
        self.assertEqual(complex_numbers.complex_conjugate(z), complex(3, -4))

    # Calculus (Numerical Differentiation & Integration)
    def test_calculus(self):
        # f(x) = x^2, f'(3) = 6
        d1 = derivative.numerical_derivative(lambda x: x**2, 3.0)
        self.assertAlmostEqual(d1, 6.0, places=4)

        # ∫[0, 2] x^2 dx = 8/3 ≈ 2.666667
        integral = derivative.numerical_integral_simpson(lambda x: x**2, 0.0, 2.0)
        self.assertAlmostEqual(integral, 8.0 / 3.0, places=4)

    # Statistics & Linear Algebra
    def test_statistics_and_vectors(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertAlmostEqual(custom_math.mean_val(data), 5.0)
        self.assertAlmostEqual(custom_math.median_val(data), 5.0)
        self.assertAlmostEqual(custom_math.std_dev(data), 2.7386127, places=4)

        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        self.assertEqual(custom_math.vector_dot(v1, v2), 32)
        self.assertEqual(custom_math.vector_cross_3d([1, 0, 0], [0, 1, 0]), [0, 0, 1])

    # Physics
    def test_physics_formulas(self):
        ke = physics.kinetic_energy(2.0, 10.0)
        self.assertAlmostEqual(ke, 100.0)
        pe = physics.gravitational_potential_energy(2.0, 10.0, g=9.8)
        self.assertAlmostEqual(pe, 196.0)

    # Comprehensive Statistics Module
    def test_statistics_module(self):
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.assertAlmostEqual(statistics.mean(data), 50.0)
        self.assertAlmostEqual(statistics.median(data), 50.0)
        self.assertEqual(statistics.mode([1, 2, 2, 3]), [2])
        self.assertAlmostEqual(statistics.geometric_mean([2, 8]), 4.0)
        self.assertAlmostEqual(statistics.harmonic_mean([2, 6]), 3.0)
        self.assertAlmostEqual(statistics.range_val(data), 80.0)
        
        q1, q2, q3 = statistics.quartiles(data)
        self.assertAlmostEqual(q1, 25.0)
        self.assertAlmostEqual(q2, 50.0)
        self.assertAlmostEqual(q3, 75.0)
        self.assertAlmostEqual(statistics.iqr(data), 50.0)

        # Bivariate and Linear Regression: y = 2*x + 1
        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]
        reg = statistics.linear_regression(x, y)
        self.assertAlmostEqual(reg["slope"], 2.0)
        self.assertAlmostEqual(reg["intercept"], 1.0)
        self.assertAlmostEqual(reg["r"], 1.0)
        self.assertAlmostEqual(reg["r_squared"], 1.0)

        # Normal PDF & CDF
        self.assertAlmostEqual(statistics.normal_pdf(0, 0, 1), 1.0 / math.sqrt(2 * math.pi), places=4)
        self.assertAlmostEqual(statistics.normal_cdf(0, 0, 1), 0.5)

        # Poisson PMF & CDF
        # P(X=0) for lambda=2: e^-2 ≈ 0.135335
        self.assertAlmostEqual(statistics.poisson_pmf(0, 2.0), math.exp(-2.0))
        self.assertAlmostEqual(statistics.exponential_cdf(1.0, 1.0), 1.0 - math.exp(-1.0))

    # Dimensional Formula Compatibility & Physical Quantities
    def test_dimensional_analysis(self):
        # Pressure / Force -> [L^-2]
        res1 = dimensions.evaluate_dimensional_expression("Pressure / Force")
        self.assertEqual(res1["dimensional_formula"], "[L^-2]")
        self.assertEqual(res1["powers"]["L"], -2)
        self.assertEqual(res1["powers"]["M"], 0)

        # Force * Distance -> [M L^2 T^-2] (Energy / Work)
        res2 = dimensions.evaluate_dimensional_expression("Force * Distance")
        self.assertEqual(res2["dimensional_formula"], "[M L^2 T^-2]")
        self.assertEqual(res2["matching_quantity"], "Work")

        # Energy / Time -> [M L^2 T^-3] (Power)
        res3 = dimensions.evaluate_dimensional_expression("Energy / Time")
        self.assertEqual(res3["dimensional_formula"], "[M L^2 T^-3]")
        self.assertEqual(res3["matching_quantity"], "Power")

        # Velocity / Time -> [L T^-2] (Acceleration)
        res4 = dimensions.evaluate_dimensional_expression("Velocity / Time")
        self.assertEqual(res4["dimensional_formula"], "[L T^-2]")
        self.assertEqual(res4["matching_quantity"], "Acceleration")

        # Dimensionless ratio: Velocity / Speed
        res5 = dimensions.evaluate_dimensional_expression("Velocity / Speed")
        self.assertTrue(res5["is_dimensionless"])
        self.assertEqual(res5["dimensional_formula"], "[1] (Dimensionless)")

if __name__ == "__main__":
    unittest.main()
