"""
Binomial theorem expansions, combinations (nCr), permutations (nPr), 
Pascal's triangle, generalized Newton binomial series, and Binomial probability distributions,
encapsulated inside BinomialClass without using the math module.
"""
from typing import Union, List, Tuple

class BinomialClass:
    """Encapsulates binomial calculations and combinatorics without using the math module."""

    # --- Pure Python Custom Math Helpers ---

    @staticmethod
    def _factorial(n: int) -> int:
        """Computes factorial without importing math."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        res = 1
        for i in range(2, n + 1):
            res *= i
        return res

    @classmethod
    def _comb(cls, n: int, r: int) -> int:
        """Computes combinations C(n, r) without importing math."""
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        if r > n // 2:
            r = n - r
        
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    @staticmethod
    def _perm(n: int, r: int) -> int:
        """Computes permutations P(n, r) without importing math."""
        if r < 0 or r > n:
            return 0
        res = 1
        for i in range(n, n - r, -1):
            res *= i
        return res

    @staticmethod
    def _sqrt(x: float) -> float:
        """Computes square root using Newton-Raphson method without importing math."""
        if x < 0:
            raise ValueError("Square root not defined for negative numbers")
        if x == 0.0:
            return 0.0
        
        guess = x / 2.0
        while True:
            better_guess = (guess + x / guess) / 2.0
            if abs(better_guess - guess) < 1e-12:
                return better_guess
            guess = better_guess

    # --- Core Methods ---

    def nCr(self, n: Union[int, float], r: int) -> Union[int, float]:
        """
        Combinations / Binomial Coefficient: n choose r.
        Supports integer n as well as generalized real n.
        """
        if isinstance(n, int) and n >= 0:
            r = int(r)
            if r < 0 or r > n:
                return 0
            return self._comb(n, r)
        else:
            return self.generalized_binomial_coeff(n, int(r))

    def nPr(self, n: int, r: int) -> int:
        """Permutations: P(n, r) = n! / (n - r)!."""
        n = int(n)
        r = int(r)
        if r < 0 or r > n:
            return 0
        return self._perm(n, r)

    def generalized_binomial_coeff(self, alpha: float, k: int) -> float:
        """
        Generalized Newton's Binomial Coefficient for any real exponent alpha and integer k >= 0:
        binom(alpha, k) = (alpha * (alpha - 1) * ... * (alpha - k + 1)) / k!
        """
        k = int(k)
        if k < 0:
            return 0.0
        if k == 0:
            return 1.0
        
        numerator = 1.0
        for i in range(k):
            numerator *= (alpha - i)
        return numerator / self._factorial(k)

    def binomial_theorem_term(self, n: int, k: int, a: float, b: float) -> float:
        """
        Computes a single k-th term in the Binomial expansion of (a + b)^n:
        Term_k = C(n, k) * a^(n - k) * b^k
        """
        n = int(n)
        k = int(k)
        if k < 0 or k > n:
            return 0.0
        coeff = self._comb(n, k)
        return coeff * (a ** (n - k)) * (b ** k)

    def binomial_expansion(self, n: int, a: float, b: float) -> dict:
        """
        Calculates the complete Binomial Theorem expansion for (a + b)^n:
        (a + b)^n = sum_{k=0}^n [ C(n, k) * a^(n-k) * b^k ]
        """
        n = int(n)
        if n < 0:
            raise ValueError("Standard polynomial binomial expansion requires non-negative integer n")
        
        terms = []
        coefficients = []
        total = 0.0
        
        for k in range(n + 1):
            c = self._comb(n, k)
            coefficients.append(c)
            term_val = c * (a ** (n - k)) * (b ** k)
            terms.append(term_val)
            total += term_val
            
        return {
            "n": n,
            "a": a,
            "b": b,
            "sum": total,
            "direct_power": (a + b) ** n,
            "coefficients": coefficients,
            "terms": terms
        }

    def binomial_expansion_str(self, n: int, a_sym: str = "a", b_sym: str = "b") -> str:
        """
        Generates the symbolic algebraic representation of (a + b)^n using the Binomial Theorem.
        Example for n=3: "a^3 + 3*a^2*b + 3*a*b^2 + b^3"
        """
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative integer")
        if n == 0:
            return "1"
        
        terms = []
        for k in range(n + 1):
            coeff = self._comb(n, k)
            a_pow = n - k
            b_pow = k
            
            parts = []
            if coeff != 1 or (a_pow == 0 and b_pow == 0):
                parts.append(str(coeff))
                
            if a_pow == 1:
                parts.append(a_sym)
            elif a_pow > 1:
                parts.append(f"{a_sym}^{a_pow}")
                
            if b_pow == 1:
                parts.append(b_sym)
            elif b_pow > 1:
                parts.append(f"{b_sym}^{b_pow}")
                
            term_str = "*".join(parts) if parts else "1"
            terms.append(term_str)
            
        return " + ".join(terms)

    def binomial_series_infinite(self, x: float, alpha: float, num_terms: int = 10) -> dict:
        """
        Newton's Generalized Binomial Series for (1 + x)^alpha where |x| < 1 or real alpha:
        (1 + x)^alpha = sum_{k=0}^inf [ binom(alpha, k) * x^k ]
        """
        num_terms = int(num_terms)
        terms = []
        total = 0.0
        
        for k in range(num_terms):
            coeff = self.generalized_binomial_coeff(alpha, k)
            term = coeff * (x ** k)
            terms.append(term)
            total += term
            
        return {
            "x": x,
            "alpha": alpha,
            "terms_computed": num_terms,
            "approximation": total,
            "terms": terms
        }

    def binomial_pmf(self, n: int, k: int, p: float) -> float:
        """
        Binomial Probability Mass Function:
        P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
        """
        n = int(n)
        k = int(k)
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be between 0 and 1")
        if k < 0 or k > n:
            return 0.0
        return self._comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))

    def binomial_cdf(self, n: int, k: int, p: float) -> float:
        """
        Cumulative Binomial Distribution:
        P(X <= k) = sum_{i=0}^k [ P(X = i) ]
        """
        n = int(n)
        k = int(k)
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be between 0 and 1")
        if k < 0:
            return 0.0
        if k >= n:
            return 1.0
        
        return sum(self.binomial_pmf(n, i, p) for i in range(k + 1))

    def binomial_stats(self, n: int, p: float) -> dict:
        """
        Computes statistical properties of Binomial distribution B(n, p):
        """
        n = int(n)
        if not (0.0 <= p <= 1.0):
            raise ValueError("Probability p must be between 0 and 1")
        
        mean = n * p
        variance = n * p * (1.0 - p)
        std_dev = self._sqrt(variance) if variance >= 0 else 0.0
        skewness = (1.0 - 2.0 * p) / std_dev if std_dev > 0 else 0.0
        kurtosis = (1.0 - 6.0 * p * (1.0 - p)) / variance if variance > 0 else 0.0
        
        return {
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "skewness": skewness,
            "kurtosis": kurtosis
        }

    def negative_binomial_pmf(self, k: int, r: int, p: float) -> float:
        """
        Negative Binomial Distribution (number of failures k before r successes):
        P(X = k) = C(k + r - 1, k) * p^r * (1 - p)^k
        """
        k = int(k)
        r = int(r)
        if not (0.0 < p <= 1.0):
            raise ValueError("Probability p must be in (0, 1]")
        if k < 0 or r <= 0:
            return 0.0
        return self._comb(k + r - 1, k) * (p ** r) * ((1.0 - p) ** k)

    def multinomial_coeff(self, n: int, *ks: int) -> int:
        """Multinomial coefficient: n! / (k1! * k2! * ... * km!)."""
        if sum(ks) != n:
            raise ValueError("Sum of k values must equal n")
        res = self._factorial(n)
        for k in ks:
            res //= self._factorial(k)
        return res

    def catalan_number(self, n: int) -> int:
        """Catalan number C_n = (2n)! / ((n+1)! * n!)."""
        n = int(n)
        if n < 0:
            raise ValueError("Catalan numbers defined for n >= 0")
        return self._comb(2 * n, n) // (n + 1)

    def pascals_triangle_row(self, n: int) -> list[int]:
        """Return n-th row of Pascal's triangle (0-indexed)."""
        return [self._comb(n, k) for k in range(n + 1)]
        
