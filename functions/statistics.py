"""
Comprehensive Statistics, Probability Distributions, and Regression Analysis Module.
Pure Python implementations of descriptive statistics, inferential tests, 
and continuous/discrete probability density and cumulative distribution functions.
"""
import math
from typing import List, Tuple, Dict, Any, Union

# ==========================================
# 1. DESCRIPTIVE STATISTICS
# ==========================================

def mean(data: List[Union[int, float]]) -> float:
    """Calculates the arithmetic mean: sum(x) / N."""
    if not data:
        raise ValueError("Data list cannot be empty")
    return float(sum(data) / len(data))

def median(data: List[Union[int, float]]) -> float:
    """Calculates the median (middle value) of a dataset."""
    if not data:
        raise ValueError("Data list cannot be empty")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)

def mode(data: List[Union[int, float]]) -> List[Union[int, float]]:
    """Calculates the mode(s) (most frequent value(s)) of a dataset."""
    if not data:
        raise ValueError("Data list cannot be empty")
    counts: Dict[Union[int, float], int] = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]

def geometric_mean(data: List[Union[int, float]]) -> float:
    """Calculates the geometric mean: (prod(x))^(1/N). All x must be > 0."""
    if not data:
        raise ValueError("Data list cannot be empty")
    if any(x <= 0 for x in data):
        raise ValueError("All elements must be positive for geometric mean")
    # Use log sum for numerical stability
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))

def harmonic_mean(data: List[Union[int, float]]) -> float:
    """Calculates the harmonic mean: N / sum(1/x). All x must be > 0."""
    if not data:
        raise ValueError("Data list cannot be empty")
    if any(x <= 0 for x in data):
        raise ValueError("All elements must be positive for harmonic mean")
    return float(len(data) / sum(1.0 / x for x in data))

def variance(data: List[Union[int, float]], sample: bool = True) -> float:
    """
    Calculates sample variance (N-1 denominator) or population variance (N denominator).
    """
    n = len(data)
    if n < 2 and sample:
        raise ValueError("Sample variance requires at least 2 data points")
    if n == 0:
        raise ValueError("Data list cannot be empty")
    
    m = mean(data)
    ss = sum((x - m) ** 2 for x in data)
    denom = (n - 1) if sample else n
    return float(ss / denom)

def std_dev(data: List[Union[int, float]], sample: bool = True) -> float:
    """Calculates standard deviation: sqrt(variance)."""
    return math.sqrt(variance(data, sample=sample))

def range_val(data: List[Union[int, float]]) -> float:
    """Calculates the range: max(data) - min(data)."""
    if not data:
        raise ValueError("Data list cannot be empty")
    return float(max(data) - min(data))

def quartiles(data: List[Union[int, float]]) -> Tuple[float, float, float]:
    """
    Calculates the first quartile (Q1), median (Q2), and third quartile (Q3).
    """
    if len(data) < 4:
        raise ValueError("Quartiles require at least 4 data points")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    
    q2 = median(s)
    if n % 2 == 0:
        q1 = median(s[:mid])
        q3 = median(s[mid:])
    else:
        q1 = median(s[:mid])
        q3 = median(s[mid + 1:])
        
    return (q1, q2, q3)

def iqr(data: List[Union[int, float]]) -> float:
    """Calculates the Interquartile Range (IQR = Q3 - Q1)."""
    q1, _, q3 = quartiles(data)
    return float(q3 - q1)

def skewness(data: List[Union[int, float]]) -> float:
    """
    Sample skewness (Fisher-Pearson coefficient of skewness).
    g1 = (n / ((n-1)(n-2))) * sum(((x - mean)/s)^3)
    """
    n = len(data)
    if n < 3:
        raise ValueError("Skewness calculation requires at least 3 data points")
    m = mean(data)
    s = std_dev(data, sample=True)
    if s == 0:
        return 0.0
    
    sum_cubed = sum(((x - m) / s) ** 3 for x in data)
    factor = n / ((n - 1) * (n - 2))
    return float(factor * sum_cubed)

def kurtosis(data: List[Union[int, float]], excess: bool = True) -> float:
    """
    Sample excess kurtosis (relative to standard normal kurtosis of 3).
    """
    n = len(data)
    if n < 4:
        raise ValueError("Kurtosis calculation requires at least 4 data points")
    m = mean(data)
    s = std_dev(data, sample=True)
    if s == 0:
        return 0.0
    
    sum_fourth = sum(((x - m) / s) ** 4 for x in data)
    factor1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    factor2 = (3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))
    
    kurt = factor1 * sum_fourth - factor2
    return float(kurt if excess else kurt + 3.0)

def z_scores(data: List[Union[int, float]]) -> List[float]:
    """Calculates the standardized z-scores: z = (x - mean) / std_dev."""
    m = mean(data)
    s = std_dev(data, sample=True)
    if s == 0:
        return [0.0 for _ in data]
    return [(x - m) / s for x in data]

def describe(data: List[Union[int, float]]) -> Dict[str, Any]:
    """
    Returns a comprehensive statistical summary dictionary of the dataset.
    """
    q1, q2, q3 = quartiles(data) if len(data) >= 4 else (None, median(data), None)
    return {
        "count": len(data),
        "mean": mean(data),
        "median": median(data),
        "mode": mode(data),
        "min": min(data),
        "max": max(data),
        "range": range_val(data),
        "q1": q1,
        "q3": q3,
        "iqr": (q3 - q1) if (q1 is not None and q3 is not None) else None,
        "variance": variance(data, sample=True) if len(data) > 1 else 0.0,
        "std_dev": std_dev(data, sample=True) if len(data) > 1 else 0.0,
        "skewness": skewness(data) if len(data) >= 3 else None,
        "kurtosis": kurtosis(data) if len(data) >= 4 else None,
    }


# ==========================================
# 2. BIVARIATE & REGRESSION ANALYSIS
# ==========================================

def covariance(x: List[Union[int, float]], y: List[Union[int, float]], sample: bool = True) -> float:
    """Calculates covariance between two datasets X and Y."""
    if len(x) != len(y):
        raise ValueError("Datasets X and Y must have the same length")
    n = len(x)
    if n < 2 and sample:
        raise ValueError("Sample covariance requires at least 2 points")
    
    mx = mean(x)
    my = mean(y)
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    denom = (n - 1) if sample else n
    return float(cov / denom)

def correlation(x: List[Union[int, float]], y: List[Union[int, float]]) -> float:
    """
    Calculates Pearson's correlation coefficient r = Cov(X, Y) / (Sx * Sy).
    """
    sx = std_dev(x, sample=True)
    sy = std_dev(y, sample=True)
    if sx == 0 or sy == 0:
        return 0.0
    return float(covariance(x, y, sample=True) / (sx * sy))

def linear_regression(x: List[Union[int, float]], y: List[Union[int, float]]) -> Dict[str, Any]:
    """
    Calculates ordinary least squares linear regression: y = slope * x + intercept.
    
    Returns:
    - slope (m)
    - intercept (c)
    - r (Pearson correlation)
    - r_squared (coefficient of determination)
    - equation_str: 'y = m*x + c'
    """
    if len(x) != len(y):
        raise ValueError("Datasets X and Y must have identical lengths")
    n = len(x)
    if n < 2:
        raise ValueError("Regression requires at least 2 data points")
    
    mx = mean(x)
    my = mean(y)
    
    ss_xx = sum((xi - mx) ** 2 for xi in x)
    ss_xy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    
    if ss_xx == 0:
        raise ValueError("Variance in X is zero, vertical line cannot be fitted")
    
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    r = correlation(x, y)
    r_squared = r ** 2
    
    sign_str = "+" if intercept >= 0 else "-"
    equation_str = f"y = {slope:.6g}*x {sign_str} {abs(intercept):.6g}"
    
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "r_squared": float(r_squared),
        "equation": equation_str
    }


# ==========================================
# 3. PROBABILITY DISTRIBUTIONS
# ==========================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Normal Probability Density Function:
    f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
    """
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be positive")
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * (((x - mu) / sigma) ** 2)
    return float(coeff * math.exp(exponent))

def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Cumulative Normal Distribution Function:
    Phi(x) = 0.5 * [1 + erf((x - mu) / (sigma * sqrt(2)))]
    """
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be positive")
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return float(0.5 * (1.0 + math.erf(z)))

def standard_normal_cdf(z: float) -> float:
    """Standard Normal CDF: Phi(z) with mu=0, sigma=1."""
    return normal_cdf(z, mu=0.0, sigma=1.0)

def poisson_pmf(k: int, lam: float) -> float:
    """
    Poisson Probability Mass Function:
    P(X = k) = (lambda^k * exp(-lambda)) / k!
    """
    k = int(k)
    if lam <= 0:
        raise ValueError("Rate parameter lambda must be positive")
    if k < 0:
        return 0.0
    return float((math.exp(-lam) * (lam ** k)) / math.factorial(k))

def poisson_cdf(k: int, lam: float) -> float:
    """
    Cumulative Poisson Distribution:
    P(X <= k) = sum_{i=0}^k [ (lambda^i * exp(-lambda)) / i! ]
    """
    k = int(k)
    if lam <= 0:
        raise ValueError("Rate parameter lambda must be positive")
    if k < 0:
        return 0.0
    return float(sum(poisson_pmf(i, lam) for i in range(k + 1)))

def exponential_pdf(x: float, rate: float) -> float:
    """
    Exponential Probability Density Function:
    f(x) = lambda * exp(-lambda * x) for x >= 0
    """
    if rate <= 0:
        raise ValueError("Rate parameter lambda must be positive")
    if x < 0:
        return 0.0
    return float(rate * math.exp(-rate * x))

def exponential_cdf(x: float, rate: float) -> float:
    """
    Exponential Cumulative Distribution Function:
    F(x) = 1 - exp(-lambda * x) for x >= 0
    """
    if rate <= 0:
        raise ValueError("Rate parameter lambda must be positive")
    if x < 0:
        return 0.0
    return float(1.0 - math.exp(-rate * x))

def uniform_pdf(x: float, a: float, b: float) -> float:
    """Uniform Probability Density Function on [a, b]."""
    if b <= a:
        raise ValueError("Upper bound b must be strictly greater than lower bound a")
    if a <= x <= b:
        return float(1.0 / (b - a))
    return 0.0

def uniform_cdf(x: float, a: float, b: float) -> float:
    """Uniform Cumulative Distribution Function on [a, b]."""
    if b <= a:
        raise ValueError("Upper bound b must be strictly greater than lower bound a")
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return float((x - a) / (b - a))

def t_pdf(x: float, df: int) -> float:
    """
    Student's t-distribution Probability Density Function:
    f(t) = Gamma((df+1)/2) / (sqrt(df*pi) * Gamma(df/2)) * (1 + t^2/df)^(-(df+1)/2)
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    coeff = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))
    power = -(df + 1) / 2.0
    return float(coeff * ((1.0 + (x ** 2) / df) ** power))
