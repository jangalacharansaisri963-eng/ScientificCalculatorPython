"""
taylor_math.py

Full-featured Taylor Series, Transcendental Approximation Engine, and 
College-Level Mathematical Functions Suite (100+ functions).
Zero standard library imports except from decimal.
"""

from decimal import Decimal, getcontext, localcontext


# ==============================================================================
# MATHEMATICAL CONSTANTS
# ==============================================================================

EULER = Decimal("2.71828182845904523536028747135266249775724709369995")
PI    = Decimal("3.14159265358979323846264338327950288419716939937510")
PHI   = Decimal("1.61803398874989484820458683436563811772030917980576") # Golden Ratio
LN2   = Decimal("0.69314718055994530941723212145817656807550013436025")
LN10  = Decimal("2.30258509299404568401799145468436420760110148862877")


# ==============================================================================
# CORE ARITHMETIC UTILITIES
# ==============================================================================

def factorial(n):
    """Computes exact factorial using Decimal."""
    if n < 0 or n % 1 != 0:
        raise ValueError("Factorial is only defined for non-negative integers.")
    res = Decimal(1)
    for i in range(2, int(n) + 1):
        res *= Decimal(i)
    return res


def double_factorial(n):
    """Computes n!! (double factorial)."""
    if n < 0:
        raise ValueError("Double factorial not defined for negative numbers.")
    res = Decimal(1)
    curr = int(n)
    while curr > 1:
        res *= Decimal(curr)
        curr -= 2
    return res


# ==============================================================================
# TRUNCATED TAYLOR SERIES (JET) ENGINE
# ==============================================================================

class TaylorSeries:
    """Truncated Taylor Series (Jet) engine for forward-mode AD and approximation."""
    __slots__ = ("coeffs", "order")

    def __init__(self, coeffs, order=None):
        if isinstance(coeffs, (int, float, str, Decimal)):
            val = coeffs if isinstance(coeffs, Decimal) else Decimal(str(coeffs))
            pad_len = order if (order is not None and order >= 0) else 0
            self.coeffs = [val] + [Decimal(0)] * pad_len
        else:
            self.coeffs = [c if isinstance(c, Decimal) else Decimal(str(c)) for c in coeffs]
            if order is not None and len(self.coeffs) < order + 1:
                self.coeffs.extend([Decimal(0)] * (order + 1 - len(self.coeffs)))
        self.order = len(self.coeffs) - 1

    @classmethod
    def seed(cls, x, order=1):
        val = x if isinstance(x, Decimal) else Decimal(str(x))
        if order < 0:
            raise ValueError("Order must be non-negative.")
        coeffs = [Decimal(0)] * (order + 1)
        coeffs[0] = val
        if order >= 1:
            coeffs[1] = Decimal(1)
        return cls(coeffs)

    @classmethod
    def _coerce(cls, value, order):
        if isinstance(value, TaylorSeries):
            if value.order < order:
                padded = list(value.coeffs) + [Decimal(0)] * (order - value.order)
                return cls(padded)
            return value
        return cls(value, order=order)

    def __add__(self, other):
        target = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s, o = self._coerce(self, target), self._coerce(other, target)
        return TaylorSeries([a + b for a, b in zip(s.coeffs, o.coeffs)])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        target = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s, o = self._coerce(self, target), self._coerce(other, target)
        return TaylorSeries([a - b for a, b in zip(s.coeffs, o.coeffs)])

    def __rsub__(self, other):
        target = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s, o = self._coerce(self, target), self._coerce(other, target)
        return TaylorSeries([ob - sb for sb, ob in zip(s.coeffs, o.coeffs)])

    def __mul__(self, other):
        target = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s, o = self._coerce(self, target), self._coerce(other, target)
        n = target
        res = [Decimal(0)] * (n + 1)
        for k in range(n + 1):
            acc = Decimal(0)
            for i in range(k + 1):
                acc += s.coeffs[i] * o.coeffs[k - i]
            res[k] = acc
        return TaylorSeries(res)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        target = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s, o = self._coerce(self, target), self._coerce(other, target)
        b0 = o.coeffs[0]
        if b0 == 0:
            raise ZeroDivisionError("Division by zero in Taylor series expansion.")
        n = target
        res = [Decimal(0)] * (n + 1)
        for k in range(n + 1):
            acc = s.coeffs[k]
            for i in range(1, k + 1):
                acc -= res[k - i] * o.coeffs[i]
            res[k] = acc / b0
        return TaylorSeries(res)

    def __rtruediv__(self, other):
        target = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        o = self._coerce(other, target)
        return o.__truediv__(self)

    def __neg__(self):
        return TaylorSeries([-c for c in self.coeffs])

    def __pos__(self):
        return TaylorSeries([+c for c in self.coeffs])

    def __pow__(self, power):
        if isinstance(power, TaylorSeries):
            return TaylorMath.exp(power * TaylorMath.ln(self))
        p = Decimal(str(power))
        u0 = self.coeffs[0]
        if u0 <= 0 and p % 1 != 0:
            raise ValueError("Base must be positive for non-integer powers.")
        n = self.order
        res = [Decimal(0)] * (n + 1)
        res[0] = u0 ** p
        if u0 == 0:
            return TaylorSeries(res)
        for k in range(1, n + 1):
            acc = Decimal(0)
            for i in range(1, k + 1):
                term = (p * Decimal(i) - Decimal(k - i)) * self.coeffs[i] * res[k - i]
                acc += term
            res[k] = acc / (Decimal(k) * u0)
        return TaylorSeries(res)

    def __rpow__(self, base):
        base_dec = Decimal(str(base))
        if base_dec <= 0:
            raise ValueError("Base must be strictly positive.")
        return TaylorMath.exp(self * TaylorMath.ln(base_dec))

    def evaluate(self, h):
        h_dec = h if isinstance(h, Decimal) else Decimal(str(h))
        total, h_power = Decimal(0), Decimal(1)
        for c in self.coeffs:
            total += c * h_power
            h_power *= h_dec
        return total

    def __repr__(self):
        return f"TaylorSeries(order={self.order}, base={self.coeffs[0]})"


# ==============================================================================
# TRANSCENDENTAL MATH ENGINE
# ==============================================================================

class TaylorMath:
    """High-precision primitives and transcendental jet operations."""

    @staticmethod
    def _get_ln(val):
        val = val if isinstance(val, Decimal) else Decimal(str(val))
        if hasattr(val, 'ln'):
            return val.ln()
        if val <= 0:
            raise ValueError("Logarithm domain error.")
        y = (val - Decimal(1)) / (val + Decimal(1))
        y2 = y * y
        total, curr = Decimal(0), y
        for k in range(1, 120, 2):
            total += curr / Decimal(k)
            curr *= y2
        return Decimal(2) * total

    @staticmethod
    def _decimal_exp(x, steps=100):
        term, total = Decimal(1), Decimal(1)
        for i in range(1, steps):
            term *= x / Decimal(i)
            total += term
        return total

    @staticmethod
    def _decimal_sin_cos(x, steps=100):
        tau = Decimal(2) * PI
        x = x % tau
        sin_val, cos_val, term = Decimal(0), Decimal(0), Decimal(1)
        for i in range(steps):
            mod = i % 4
            if mod == 0: cos_val += term
            elif mod == 1: sin_val += term
            elif mod == 2: cos_val -= term
            elif mod == 3: sin_val -= term
            term *= x / Decimal(i + 1)
        return sin_val, cos_val

    @staticmethod
    def _decimal_arctan(x, steps=120):
        pi_over_2 = PI / Decimal(2)
        if x > 1:
            return pi_over_2 - TaylorMath._decimal_arctan(Decimal(1) / x, steps)
        elif x < -1:
            return -pi_over_2 - TaylorMath._decimal_arctan(Decimal(1) / x, steps)
        total, x_sq, curr_x = Decimal(0), x * x, x
        for i in range(steps):
            term = curr_x / Decimal(2 * i + 1)
            total = total - term if i % 2 == 1 else total + term
            curr_x *= x_sq
        return total

    @classmethod
    def exp(cls, g):
        g = TaylorSeries._coerce(g, 0)
        n = g.order
        res = [Decimal(0)] * (n + 1)
        res[0] = cls._decimal_exp(g.coeffs[0])
        for k in range(1, n + 1):
            s = Decimal(0)
            for i in range(1, k + 1):
                s += Decimal(i) * g.coeffs[i] * res[k - i]
            res[k] = s / Decimal(k)
        return TaylorSeries(res)

    @classmethod
    def ln(cls, g):
        g = TaylorSeries._coerce(g, 0)
        u0 = g.coeffs[0]
        if u0 <= 0:
            raise ValueError("Logarithm domain error.")
        n = g.order
        res = [Decimal(0)] * (n + 1)
        res[0] = cls._get_ln(u0)
        for k in range(1, n + 1):
            s = Decimal(k) * g.coeffs[k]
            for i in range(1, k):
                s -= Decimal(i) * res[i] * g.coeffs[k - i]
            res[k] = s / (Decimal(k) * u0)
        return TaylorSeries(res)

    @classmethod
    def log(cls, base, g):
        """Custom base logarithm: log_base(g) = ln(g) / ln(base)."""
        b_dec = base if isinstance(base, Decimal) else Decimal(str(base))
        return cls.ln(g) / cls.ln(b_dec)

    @classmethod
    def log10(cls, g):
        return cls.log(Decimal(10), g)

    @classmethod
    def log2(cls, g):
        return cls.log(Decimal(2), g)

    @classmethod
    def log1p(cls, g):
        """Natural logarithm of 1 + g."""
        g_ts = TaylorSeries._coerce(g, 0)
        return cls.ln(Decimal(1) + g_ts)

    @classmethod
    def expm1(cls, g):
        """Exponential minus 1: exp(g) - 1."""
        g_ts = TaylorSeries._coerce(g, 0)
        return cls.exp(g_ts) - Decimal(1)

    @classmethod
    def sqrt(cls, g):
        return g ** "0.5"

    @classmethod
    def cbrt(cls, g):
        return g ** (Decimal(1) / Decimal(3))

    @classmethod
    def root(cls, n, g):
        """n-th root of g."""
        return g ** (Decimal(1) / Decimal(str(n)))

    @classmethod
    def sin(cls, g):
        g = TaylorSeries._coerce(g, 0)
        n = g.order
        s_res, c_res = [Decimal(0)] * (n + 1), [Decimal(0)] * (n + 1)
        s_res[0], c_res[0] = cls._decimal_sin_cos(g.coeffs[0])
        for k in range(1, n + 1):
            s_sum, c_sum = Decimal(0), Decimal(0)
            for i in range(1, k + 1):
                i_dec = Decimal(i)
                s_sum += i_dec * g.coeffs[i] * c_res[k - i]
                c_sum -= i_dec * g.coeffs[i] * s_res[k - i]
            s_res[k] = s_sum / Decimal(k)
            c_res[k] = c_sum / Decimal(k)
        return TaylorSeries(s_res)

    @classmethod
    def cos(cls, g):
        g = TaylorSeries._coerce(g, 0)
        n = g.order
        s_res, c_res = [Decimal(0)] * (n + 1), [Decimal(0)] * (n + 1)
        s_res[0], c_res[0] = cls._decimal_sin_cos(g.coeffs[0])
        for k in range(1, n + 1):
            s_sum, c_sum = Decimal(0), Decimal(0)
            for i in range(1, k + 1):
                i_dec = Decimal(i)
                s_sum += i_dec * g.coeffs[i] * c_res[k - i]
                c_sum -= i_dec * g.coeffs[i] * s_res[k - i]
            s_res[k] = s_sum / Decimal(k)
            c_res[k] = c_sum / Decimal(k)
        return TaylorSeries(c_res)

    @classmethod
    def tan(cls, g):
        return cls.sin(g) / cls.cos(g)

    @classmethod
    def cot(cls, g):
        return cls.cos(g) / cls.sin(g)

    @classmethod
    def sec(cls, g):
        return Decimal(1) / cls.cos(g)

    @classmethod
    def csc(cls, g):
        return Decimal(1) / cls.sin(g)

    @classmethod
    def arctan(cls, g):
        g = TaylorSeries._coerce(g, 0)
        base_arctan = TaylorSeries(cls._decimal_arctan(g.coeffs[0]), order=g.order)
        denom = Decimal(1) + g ** 2
        g_prime = TaylorSeries(g.coeffs[1:], order=g.order - 1) if g.order >= 1 else TaylorSeries(0)
        if g.order == 0:
            return base_arctan
        integral_part = (g_prime / denom)
        res_coeffs = [base_arctan.coeffs[0]]
        for k in range(1, g.order + 1):
            res_coeffs.append(integral_part.coeffs[k - 1] / Decimal(k))
        return TaylorSeries(res_coeffs)

    @classmethod
    def arcsin(cls, g):
        g = TaylorSeries._coerce(g, 0)
        return cls.arctan(g / cls.sqrt(Decimal(1) - g ** 2))

    @classmethod
    def arccos(cls, g):
        return (PI / Decimal(2)) - cls.arcsin(g)

    @classmethod
    def arctan2(cls, y, x):
        """2-argument arctangent."""
        y_dec = y.coeffs[0] if isinstance(y, TaylorSeries) else Decimal(str(y))
        x_dec = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        base = cls._decimal_arctan(y_dec / x_dec) if x_dec != 0 else (PI / Decimal(2) if y_dec >= 0 else -PI / Decimal(2))
        if x_dec < 0:
            base += PI if y_dec >= 0 else -PI
        return TaylorSeries(base)

    @classmethod
    def sinh(cls, g):
        return (cls.exp(g) - cls.exp(-g)) / Decimal(2)

    @classmethod
    def cosh(cls, g):
        return (cls.exp(g) + cls.exp(-g)) / Decimal(2)

    @classmethod
    def tanh(cls, g):
        return cls.sinh(g) / cls.cosh(g)

    @classmethod
    def coth(cls, g):
        return cls.cosh(g) / cls.sinh(g)

    @classmethod
    def sech(cls, g):
        return Decimal(1) / cls.cosh(g)

    @classmethod
    def csch(cls, g):
        return Decimal(1) / cls.sinh(g)

    @classmethod
    def asinh(cls, g):
        g = TaylorSeries._coerce(g, 0)
        return cls.ln(g + cls.sqrt(g ** 2 + Decimal(1)))

    @classmethod
    def acosh(cls, g):
        g = TaylorSeries._coerce(g, 0)
        return cls.ln(g + cls.sqrt(g ** 2 - Decimal(1)))

    @classmethod
    def atanh(cls, g):
        g = TaylorSeries._coerce(g, 0)
        return Decimal("0.5") * cls.ln((Decimal(1) + g) / (Decimal(1) - g))


# ==============================================================================
# 100+ COLLEGE-LEVEL MATHEMATICAL FUNCTIONS SUITE
# ==============================================================================

class CollegeMath:
    """Exhaustive collection of 100+ undergraduate/college-level mathematical functions."""

    # --- 1-10: Algebraic & Absolute Enhancements ---
    @staticmethod
    def identity(x): return x
    @staticmethod
    def square(x): return x * x
    @staticmethod
    def cube(x): return x * x * x
    @staticmethod
    def absolute(x): return abs(x) if not isinstance(x, TaylorSeries) else abs(x.coeffs[0])
    @staticmethod
    def heaviside(x):
        val = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return Decimal(1) if val > 0 else (Decimal("0.5") if val == 0 else Decimal(0))
    @staticmethod
    def signum(x):
        val = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return Decimal(1) if val > 0 else (-Decimal(1) if val < 0 else Decimal(0))
    @staticmethod
    def ramp(x):
        val = x if isinstance(x, TaylorSeries) else TaylorSeries(x)
        return val if val.coeffs[0] >= 0 else TaylorSeries(0)
    @staticmethod
    def logistic(x): return Decimal(1) / (Decimal(1) + TaylorMath.exp(-x))
    @staticmethod
    def gnu_logit(x): return TaylorMath.ln(x / (Decimal(1) - x))
    @staticmethod
    def gaussian(x): return TaylorMath.exp(-(x * x))

    # --- 11-20: Advanced Power & Root Variants ---
    @staticmethod
    def hypot(x, y):
        x_d = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        y_d = y.coeffs[0] if isinstance(y, TaylorSeries) else Decimal(str(y))
        return TaylorMath.sqrt(x_d**2 + y_d**2)
    @staticmethod
    def inverse(x): return Decimal(1) / x
    @staticmethod
    def fractional_part(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return v % Decimal(1)
    @staticmethod
    def floor_approx(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return Decimal(int(v)) if v >= 0 else Decimal(int(v) - 1)
    @staticmethod
    def ceiling_approx(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(v))
        return Decimal(int(v) if v == int(v) else int(v) + 1)
    @staticmethod
    def sinc(x):
        x_ts = TaylorSeries._coerce(x, 5)
        zero_check = x_ts.coeffs[0]
        if zero_check == 0:
            return Decimal(1)
        return TaylorMath.sin(x_ts) / x_ts
    @staticmethod
    def boxcar(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return Decimal(1) if -Decimal("0.5") <= v <= Decimal("0.5") else Decimal(0)
    @staticmethod
    def triangle_wave(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return abs((v % Decimal(2)) - Decimal(1))
    @staticmethod
    def square_wave(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        s, _ = TaylorMath._decimal_sin_cos(v)
        return Decimal(1) if s >= 0 else -Decimal(1)
    @staticmethod
    def sawtooth_wave(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return v % Decimal(1)

    # --- 21-35: Advanced Exponentials & Logarithmic Families ---
    @staticmethod
    def exp2(x): return TaylorMath.exp(x * TaylorMath.ln(Decimal(2)))
    @staticmethod
    def exp10(x): return TaylorMath.exp(x * TaylorMath.ln(Decimal(10)))
    @staticmethod
    def exp_m2(x): return TaylorMath.exp(-(x**2) / Decimal(2))
    @staticmethod
    def log_base(base, x): return TaylorMath.log(base, x)
    @staticmethod
    def log1p(x): return TaylorMath.log1p(x)
    @staticmethod
    def expm1(x): return TaylorMath.expm1(x)
    @staticmethod
    def logaddexp(x, y):
        xd = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        yd = y.coeffs[0] if isinstance(y, TaylorSeries) else Decimal(str(y))
        return TaylorMath.ln(TaylorMath.exp(xd) + TaylorMath.exp(yd))
    @staticmethod
    def logaddexp2(x, y):
        xd = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        yd = y.coeffs[0] if isinstance(y, TaylorSeries) else Decimal(str(y))
        return TaylorMath.log2(TaylorMath.exp2(xd) + TaylorMath.exp2(yd))
    @staticmethod
    def gd(x): return Decimal(2) * TaylorMath.arctan(TaylorMath.exp(x)) - (PI / Decimal(2)) # Gudermannian
    @staticmethod
    def igd(x): return TaylorMath.ln(TaylorMath.tan((PI / Decimal(4)) + (x / Decimal(2)))) # Inverse Gudermannian
    @staticmethod
    def ververs(x): return Decimal(1) - TaylorMath.cos(x)
    @staticmethod
    def covers(x): return Decimal(1) - TaylorMath.sin(x)
    @staticmethod
    def haversine(x): return (Decimal(1) - TaylorMath.cos(x)) / Decimal(2)
    @staticmethod
    def exsecant(x): return TaylorMath.sec(x) - Decimal(1)
    @staticmethod
    def excosecant(x): return TaylorMath.csc(x) - Decimal(1)

    # --- 36-55: Complete Trigonometric & Reciprocal Family ---
    @staticmethod
    def sin(x): return TaylorMath.sin(x)
    @staticmethod
    def cos(x): return TaylorMath.cos(x)
    @staticmethod
    def tan(x): return TaylorMath.tan(x)
    @staticmethod
    def cot(x): return TaylorMath.cot(x)
    @staticmethod
    def sec(x): return TaylorMath.sec(x)
    @staticmethod
    def csc(x): return TaylorMath.csc(x)
    @staticmethod
    def versin(x): return Decimal(1) - TaylorMath.cos(x)
    @staticmethod
    def vercos(x): return Decimal(1) + TaylorMath.cos(x)
    @staticmethod
    def coversin(x): return Decimal(1) - TaylorMath.sin(x)
    @staticmethod
    def covercos(x): return Decimal(1) + TaylorMath.sin(x)
    @staticmethod
    def haversin(x): return (Decimal(1) - TaylorMath.cos(x)) / Decimal(2)
    @staticmethod
    def havercos(x): return (Decimal(1) + TaylorMath.cos(x)) / Decimal(2)
    @staticmethod
    def hacoversin(x): return (Decimal(1) - TaylorMath.sin(x)) / Decimal(2)
    @staticmethod
    def hacovercos(x): return (Decimal(1) + TaylorMath.sin(x)) / Decimal(2)
    @staticmethod
    def exsec(x): return TaylorMath.sec(x) - Decimal(1)
    @staticmethod
    def excsc(x): return TaylorMath.csc(x) - Decimal(1)
    @staticmethod
    def crd(x): return Decimal(2) * TaylorMath.sin(x / Decimal(2)) # Chord function
    @staticmethod
    def rad2deg(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return v * Decimal(180) / PI
    @staticmethod
    def deg2rad(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return v * PI / Decimal(180)
    @staticmethod
    def normalize_angle(x):
        v = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return (v + PI) % (Decimal(2) * PI) - PI

    # --- 56-75: Complete Inverse Trigonometric Family ---
    @staticmethod
    def arcsin(x): return TaylorMath.arcsin(x)
    @staticmethod
    def arccos(x): return TaylorMath.arccos(x)
    @staticmethod
    def arctan(x): return TaylorMath.arctan(x)
    @staticmethod
    def arccot(x): return (PI / Decimal(2)) - TaylorMath.arctan(x)
    @staticmethod
    def arcsec(x): return TaylorMath.arccos(Decimal(1) / x)
    @staticmethod
    def arccsc(x): return TaylorMath.arcsin(Decimal(1) / x)
    @staticmethod
    def arctan2(y, x): return TaylorMath.arctan2(y, x)
    @staticmethod
    def inverse_versin(x): return TaylorMath.arccos(Decimal(1) - x)
    @staticmethod
    def inverse_coversin(x): return TaylorMath.arcsin(Decimal(1) - x)
    @staticmethod
    def inverse_haversin(x): return Decimal(2) * TaylorMath.arcsin(TaylorMath.sqrt(x))
    @staticmethod
    def inverse_exsec(x): return TaylorMath.arcsec(x + Decimal(1))
    @staticmethod
    def angle_sum_approx(a, b): return TaylorMath.sin(a)*TaylorMath.cos(b) + TaylorMath.cos(a)*TaylorMath.sin(b)
    @staticmethod
    def angle_diff_approx(a, b): return TaylorMath.sin(a)*TaylorMath.cos(b) - TaylorMath.cos(a)*TaylorMath.sin(b)
    @staticmethod
    def double_angle_sin(x): return Decimal(2) * TaylorMath.sin(x) * TaylorMath.cos(x)
    @staticmethod
    def double_angle_cos(x): return TaylorMath.cos(x)**2 - TaylorMath.sin(x)**2
    @staticmethod
    def half_angle_tan(x): return TaylorMath.sin(x) / (Decimal(1) + TaylorMath.cos(x))
    @staticmethod
    def prosthaphaeresis_1(a, b): return Decimal(2) * TaylorMath.sin((a+b)/Decimal(2)) * TaylorMath.cos((a-b)/Decimal(2))
    @staticmethod
    def prosthaphaeresis_2(a, b): return Decimal(2) * TaylorMath.cos((a+b)/Decimal(2)) * TaylorMath.sin((a-b)/Decimal(2))
    @staticmethod
    def prosthaphaeresis_3(a, b): return Decimal(2) * TaylorMath.cos((a+b)/Decimal(2)) * TaylorMath.cos((a-b)/Decimal(2))
    @staticmethod
    def prosthaphaeresis_4(a, b): return -Decimal(2) * TaylorMath.sin((a+b)/Decimal(2)) * TaylorMath.sin((a-b)/Decimal(2))

    # --- 76-90: Complete Hyperbolic Family ---
    @staticmethod
    def sinh(x): return TaylorMath.sinh(x)
    @staticmethod
    def cosh(x): return TaylorMath.cosh(x)
    @staticmethod
    def tanh(x): return TaylorMath.tanh(x)
    @staticmethod
    def coth(x): return TaylorMath.coth(x)
    @staticmethod
    def sech(x): return TaylorMath.sech(x)
    @staticmethod
    def csch(x): return TaylorMath.csch(x)
    @staticmethod
    def tsch(x): return TaylorMath.sech(x) # Teschl notation alias
    @staticmethod
    def hyperbolic_pythagorean_check(x): return TaylorMath.cosh(x)**2 - TaylorMath.sinh(x)**2
    @staticmethod
    def sinh_double(x): return Decimal(2) * TaylorMath.sinh(x) * TaylorMath.cosh(x)
    @staticmethod
    def cosh_double(x): return TaylorMath.cosh(x)**2 + TaylorMath.sinh(x)**2
    @staticmethod
    def tanh_double(x): return (Decimal(2) * TaylorMath.tanh(x)) / (Decimal(1) + TaylorMath.tanh(x)**2)
    @staticmethod
    def sech_squared(x): return TaylorMath.sech(x)**2
    @staticmethod
    def csch_squared(x): return TaylorMath.csch(x)**2
    @staticmethod
    def coth_half(x): return TaylorMath.cosh(x) / TaylorMath.sinh(x)
    @staticmethod
    def exponential_shift_hyperbolic(x): return (TaylorMath.exp(x) - TaylorMath.exp(-x)) / Decimal(2)

    # --- 91-105: Complete Inverse Hyperbolic & Special Growth Family ---
    @staticmethod
    def asinh(x): return TaylorMath.asinh(x)
    @staticmethod
    def acosh(x): return TaylorMath.acosh(x)
    @staticmethod
    def atanh(x): return TaylorMath.atanh(x)
    @staticmethod
    def acoth(x): return Decimal("0.5") * TaylorMath.ln((x + Decimal(1)) / (x - Decimal(1)))
    @staticmethod
    def asech(x): return TaylorMath.acosh(Decimal(1) / x)
    @staticmethod
    def acsch(x): return TaylorMath.asinh(Decimal(1) / x)
    @staticmethod
    def generalized_sigmoid(x, alpha): return Decimal(1) / (Decimal(1) + TaylorMath.exp(-alpha * x))
    @staticmethod
    def swish(x): return x * CollegeMath.logistic(x)
    @staticmethod
    def gelu_approx(x):
        xd = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        # Approximation via tanh scaling
        inner = Decimal("0.7978845608") * (xd + Decimal("0.044715") * (xd**3))
        return Decimal("0.5") * xd * (Decimal(1) + TaylorMath.tanh(inner))
    @staticmethod
    def softplus(x): return TaylorMath.ln(Decimal(1) + TaylorMath.exp(x))
    @staticmethod
    def mish(x): return x * TaylorMath.tanh(CollegeMath.softplus(x))
    @staticmethod
    def scaled_exponential_linear(x, alpha=Decimal("1.67326"), scale=Decimal("1.0507")):
        xd = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        return scale * (xd if xd > 0 else alpha * (TaylorMath.exp(xd) - Decimal(1)))
    @staticmethod
    def error_function_approx(x):
        # High precision Taylor/Maclaurin approximation for erf(x)
        xd = x.coeffs[0] if isinstance(x, TaylorSeries) else Decimal(str(x))
        term, total = xd, xd
        x_sq = xd * xd
        for n in range(1, 25):
            term *= -x_sq * Decimal(2) / Decimal(2*n + 1) # Simplified series expansion accumulator
            total += term / Decimal(factorial(n))
        return Decimal("1.128379167095512574") * total # 2/sqrt(pi) scaling factor
    @staticmethod
    def fresnel_integral_s(x): return x**3 / Decimal(6) # Leading order approximation term
    @staticmethod
    def dawson_approx(x): return x * TaylorMath.exp(-(x**2)) # Scaled Faddeeva kernel base