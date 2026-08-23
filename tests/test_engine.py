"""
Unit tests for CalculatorEngine, parser, precedence, and variables.
"""
import unittest
import math
from engine import CalculatorEngine
from parser import parse_and_eval

class TestEngineAndParser(unittest.TestCase):
    def setUp(self):
        self.engine = CalculatorEngine(angle_mode="RAD")

    def test_arithmetic_order_of_operations(self):
        res = self.engine.evaluate("3 + 4 * 2 / (1 - 5) ^ 2")
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["result"], 3.5)

    def test_power_and_unary_minus(self):
        res = self.engine.evaluate("-2^2")
        self.assertAlmostEqual(res["result"], -4)
        res2 = self.engine.evaluate("(-2)^2")
        self.assertAlmostEqual(res2["result"], 4)

    def test_implicit_multiplication(self):
        res = self.engine.evaluate("2pi")
        self.assertAlmostEqual(res["result"], 2 * math.pi)
        res2 = self.engine.evaluate("3(4 + 2)")
        self.assertAlmostEqual(res2["result"], 18)

    def test_factorial(self):
        res = self.engine.evaluate("5!")
        self.assertEqual(res["result"], 120)
        res2 = self.engine.evaluate("0!")
        self.assertEqual(res2["result"], 1)

    def test_variable_assignment_and_recall(self):
        self.engine.evaluate("x = 15")
        self.engine.evaluate("y = x * 2")
        res = self.engine.evaluate("x + y + ans")
        # x = 15, y = 30, ans = 30 -> 15 + 30 + 30 = 75
        self.assertEqual(res["result"], 75)

    def test_angle_mode_evaluation(self):
        engine_deg = CalculatorEngine(angle_mode="DEG")
        res_deg = engine_deg.evaluate("sin(30)")
        self.assertAlmostEqual(res_deg["result"], 0.5)

        engine_rad = CalculatorEngine(angle_mode="RAD")
        res_rad = engine_rad.evaluate("sin(pi/6)")
        self.assertAlmostEqual(res_rad["result"], 0.5)

    def test_complex_number_evaluation(self):
        res = self.engine.evaluate("(3 + 4j) * (3 - 4j)")
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["result"].real, 25.0)
        self.assertAlmostEqual(res["result"].imag, 0.0)

    def test_equation_solver_linear(self):
        res = self.engine.evaluate("2*x + 10 = 0")
        self.assertEqual(res["status"], "equation_solved")
        self.assertEqual(res["solution"]["type"], "unique_solution")
        self.assertAlmostEqual(res["solution"]["solution"], -5.0)

    def test_equation_solver_quadratic(self):
        res = self.engine.evaluate("x^2 - 7*x + 12 = 0")
        self.assertEqual(res["status"], "equation_solved")
        self.assertEqual(res["solution"]["type"], "two_real_roots")
        roots = sorted(res["solution"]["roots"])
        self.assertAlmostEqual(roots[0], 3.0)
        self.assertAlmostEqual(roots[1], 4.0)

if __name__ == "__main__":
    unittest.main()
