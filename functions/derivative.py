from decimal import Decimal, getcontext, localcontext


# ==============================================================================
# INTERNAL MATH HELPERS (ZERO IMPORTS)
# ==============================================================================

def _factorial(n):
    """Computes factorial using Decimal to avoid importing math."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers.")
    res = Decimal(1)
    for i in range(2, n + 1):
        res *= Decimal(i)
    return res


# ==============================================================================
# TRUNCATED TAYLOR SERIES ENGINE
# ==============================================================================

class TaylorSeries:
    """
    Truncated Taylor Series (Jet) engine for n-th order forward-mode 
    automatic differentiation.
    
    Represents: f(x + ε) = c[0] + c[1]*ε + c[2]*ε² + ... + c[n]*εⁿ
    where c[k] = f^(k)(x) / k!
    """
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
        """Seeds x for n-th order derivative tracking safely across all orders >= 0."""
        val = x if isinstance(x, Decimal) else Decimal(str(x))
        if order < 0:
            raise ValueError("Order must be non-negative.")
        coeffs = [Decimal(0)] * (order + 1)
        coeffs[0] = val
        if order >= 1:
            coeffs[1] = Decimal(1)  # d/dx (x) = 1
        return cls(coeffs)

    @classmethod
    def _coerce(cls, value, order):
        if isinstance(value, TaylorSeries):
            if value.order < order:
                padded = list(value.coeffs) + [Decimal(0)] * (order - value.order)
                return cls(padded)
            return value
        return cls(value, order=order)

    # --- Basic Polynomial Arithmetic ---

    def __add__(self, other):
        target_order = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s = self._coerce(self, target_order)
        o = self._coerce(other, target_order)
        return TaylorSeries([a + b for a, b in zip(s.coeffs, o.coeffs)])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        target_order = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s = self._coerce(self, target_order)
        o = self._coerce(other, target_order)
        return TaylorSeries([a - b for a, b in zip(s.coeffs, o.coeffs)])

    def __rsub__(self, other):
        target_order = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s = self._coerce(self, target_order)
        o = self._coerce(other, target_order)
        return TaylorSeries([ob - sb for sb, ob in zip(s.coeffs, o.coeffs)])

    def __mul__(self, other):
        target_order = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s = self._coerce(self, target_order)
        o = self._coerce(other, target_order)
        n = target_order
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
        target_order = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        s = self._coerce(self, target_order)
        o = self._coerce(other, target_order)
        b0 = o.coeffs[0]
        if b0 == 0:
            raise ZeroDivisionError("Division by zero in Taylor series expansion base coefficient.")
        n = target_order
        res = [Decimal(0)] * (n + 1)
        for k in range(n + 1):
            acc = s.coeffs[k]
            for i in range(1, k + 1):
                acc -= res[k - i] * o.coeffs[i]
            res[k] = acc / b0
        return TaylorSeries(res)

    def __rtruediv__(self, other):
        target_order = max(self.order, other.order) if isinstance(other, TaylorSeries) else self.order
        o = self._coerce(other, target_order)
        return o.__truediv__(self)

    def __neg__(self):
        return TaylorSeries([-c for c in self.coeffs])

    def __pos__(self):
        return TaylorSeries([+c for c in self.coeffs])

    def __pow__(self, power):
        """General Power Rule: (u(x))^p using exact Taylor recurrence."""
        if isinstance(power, TaylorSeries):
            return DualMath.exp(power * DualMath.ln(self))

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
            raise ValueError("Exponential base must be strictly positive.")
        return DualMath.exp(self * DualMath.ln(base_dec))

    def __repr__(self):
        return f"TaylorSeries(order={self.order}, base={self.coeffs[0]})"


# ==============================================================================
# EXPANDED TRANSCENDENTAL FUNCTIONS
# ==============================================================================

class DualMath:
    """High-precision math primitives with localized precision contexts."""

    PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068")

    @staticmethod
    def _get_ln(val):
        val = val if isinstance(val, Decimal) else Decimal(str(val))
        if hasattr(val, 'ln'):
            return val.ln()
        if val <= 0:
            raise ValueError("Logarithm domain error.")
        y = (val - Decimal(1)) / (val + Decimal(1))
        y2 = y * y
        total = Decimal(0)
        curr = y
        for k in range(1, 120, 2):
            total += curr / Decimal(k)
            curr *= y2
        return Decimal(2) * total

    @staticmethod
    def _decimal_exp(x, steps=100):
        term = Decimal(1)
        total = Decimal(1)
        for i in range(1, steps):
            term *= x / Decimal(i)
            total += term
        return total

    @staticmethod
    def _decimal_sin_cos(x, steps=100):
        tau = Decimal(2) * DualMath.PI
        x = x % tau

        sin_val, cos_val = Decimal(0), Decimal(0)
        term = Decimal(1)
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
        pi_over_2 = Decimal("1.570796326794896619231321691639751442098584699687552910487472296153908203143104499314017412671058534")
        if x > 1:
            return pi_over_2 - DualMath._decimal_arctan(Decimal(1) / x, steps)
        elif x < -1:
            return -pi_over_2 - DualMath._decimal_arctan(Decimal(1) / x, steps)

        total = Decimal(0)
        x_sq = x * x
        curr_x = x
        for i in range(steps):
            term = curr_x / Decimal(2 * i + 1)
            total = total - term if i % 2 == 1 else total + term
            curr_x *= x_sq
        return total

    # --- Transcendental Jet Operations ---

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
            raise ValueError("Logarithm domain error: base must be positive.")
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
    def sqrt(cls, g):
        return g ** "0.5"

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
        if g.coeffs[0] <= -1 or g.coeffs[0] >= 1:
            raise ValueError("Arcsin domain error: value must be in (-1, 1).")
        return cls.arctan(g / cls.sqrt(Decimal(1) - g ** 2))

    @classmethod
    def arccos(cls, g):
        pi_over_2 = Decimal("1.57079632679489661923132169163975")
        return pi_over_2 - cls.arcsin(g)

    @classmethod
    def sinh(cls, g):
        return (cls.exp(g) - cls.exp(-g)) / Decimal(2)

    @classmethod
    def cosh(cls, g):
        return (cls.exp(g) + cls.exp(-g)) / Decimal(2)

    @classmethod
    def tanh(cls, g):
        return cls.sinh(g) / cls.cosh(g)

    @staticmethod
    def _exp_scalar(x):
        val = x if isinstance(x, Decimal) else Decimal(str(x))
        return DualMath._decimal_exp(val)

    @staticmethod
    def _ln_scalar(x):
        val = x if isinstance(x, Decimal) else Decimal(str(x))
        return DualMath._get_ln(val)


# ==============================================================================
# UNIFIED DIFFERENTIATION ENGINE & EXTENDED UTILITIES
# ==============================================================================

def derivative(f, x, order=1, precision=50):
    """Computes exact n-th order derivative of single-variable f at x in O(n²) time."""
    with localcontext() as ctx:
        ctx.prec = precision
        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        if order == 0:
            res = f(TaylorSeries(x, order=0))
            return res.coeffs[0] if isinstance(res, TaylorSeries) else Decimal(str(res))
            
        z = TaylorSeries.seed(x, order)
        res = f(z)
        if not isinstance(res, TaylorSeries):
            return Decimal(0)
            
        fact = _factorial(order)
        return res.coeffs[order] * fact


def gradient(f, vars_list, precision=50):
    """Computes the Gradient Vector ∇f at multi-variable point [x1, x2, ...]."""
    with localcontext() as ctx:
        ctx.prec = precision
        n = len(vars_list)
        grad = []
        for i in range(n):
            ts_vars = [
                TaylorSeries.seed(x, order=1) if idx == i 
                else TaylorSeries(x, order=0)
                for idx, x in enumerate(vars_list)
            ]
            res = f(ts_vars)
            grad.append(res.coeffs[1] if isinstance(res, TaylorSeries) else Decimal(0))
        return grad


def hessian(f, vars_list, precision=50):
    """Computes the full n x n Hessian Matrix H (∂²f / ∂x_i ∂x_j)."""
    with localcontext() as ctx:
        ctx.prec = precision
        n = len(vars_list)
        H = [[Decimal(0)] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    ts_vars = [
                        TaylorSeries.seed(x, order=2) if idx == i 
                        else TaylorSeries(x, order=0)
                        for idx, x in enumerate(vars_list)
                    ]
                    res = f(ts_vars)
                    H[i][i] = (res.coeffs[2] * Decimal(2)) if isinstance(res, TaylorSeries) else Decimal(0)
                else:
                    ts_diag = [
                        TaylorSeries.seed(x, order=2) if idx in (i, j) 
                        else TaylorSeries(x, order=0)
                        for idx, x in enumerate(vars_list)
                    ]
                    res_diag = f(ts_diag)
                    d2_combined = (res_diag.coeffs[2] * Decimal(2)) if isinstance(res_diag, TaylorSeries) else Decimal(0)
                    
                    mixed = (d2_combined - H[i][i] - H[j][j]) / Decimal(2)
                    H[i][j] = mixed
                    H[j][i] = mixed
        return H


def jacobian(f_vec, vars_list, precision=50):
    """Computes the Jacobian Matrix J for vector-valued functions f: R^n -> R^m."""
    with localcontext() as ctx:
        ctx.prec = precision
        n = len(vars_list)
        m = len(f_vec(vars_list))
        J = [[Decimal(0)] * n for _ in range(m)]

        for j in range(n):
            ts_vars = [
                TaylorSeries.seed(x, order=1) if idx == j 
                else TaylorSeries(x, order=0)
                for idx, x in enumerate(vars_list)
            ]
            eval_res = f_vec(ts_vars)
            for i in range(m):
                val = eval_res[i]
                J[i][j] = val.coeffs[1] if isinstance(val, TaylorSeries) else Decimal(0)
        return J


def directional_derivative(f, vars_list, direction_vec, order=1, precision=50):
    """Computes the n-th order directional derivative D_v^n f(x) along vector v."""
    with localcontext() as ctx:
        ctx.prec = precision
        n = len(vars_list)
        if len(direction_vec) != n:
            raise ValueError("Direction vector length must match variables length.")

        ts_vars = []
        for x, v in zip(vars_list, direction_vec):
            v_dec = v if isinstance(v, Decimal) else Decimal(str(v))
            x_dec = x if isinstance(x, Decimal) else Decimal(str(x))
            coeffs = [x_dec] + [Decimal(0)] * order
            if order >= 1:
                coeffs[1] = v_dec
            ts_vars.append(TaylorSeries(coeffs))

        res = f(ts_vars)
        if not isinstance(res, TaylorSeries):
            return Decimal(0)

        fact = _factorial(order)
        return res.coeffs[order] * fact


def taylor_coefficients(f, x, order=5, precision=50):
    """Extracts raw normalized Taylor coefficients [c0, c1, ..., cn] for f around x."""
    with localcontext() as ctx:
        ctx.prec = precision
        z = TaylorSeries.seed(x, order=order)
        res = f(z)
        if isinstance(res, TaylorSeries):
            return res.coeffs
        return [Decimal(str(res))] + [Decimal(0)] * order


