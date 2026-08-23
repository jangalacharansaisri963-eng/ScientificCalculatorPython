# --- 1. CORE TYPE & SHAPE UTILITIES ---

class ndarray:
    """A basic array object representing multidimensional data from scratch."""
    def __init__(self, data, shape=None):
        if shape is None:
            self.data = []
            self._shape = self._infer_shape_and_flatten(data, self.data)
        else:
            self.data = []
            for x in data:
                self.data.append(x)
            shape_list = []
            for s in shape:
                shape_list.append(s)
            self._shape = tuple(shape_list)
            
    def _infer_shape_and_flatten(self, lst, flat):
        if not isinstance(lst, list):
            flat.append(lst)
            return ()
        if not lst:
            return (0,)
        inner_shape = self._infer_shape_and_flatten(lst[0], flat)
        for item in lst[1:]:
            self._infer_shape_and_flatten(item, flat)
        return (len(lst),) + inner_shape

    @property
    def shape(self): return self._shape # Attribute 1
    
    @property
    def ndim(self): # Attribute 2
        count = 0
        for _ in self._shape:
            count += 1
        return count
    
    @property
    def size(self): # Attribute 3
        p = 1
        for x in self._shape: 
            p *= x
        return p

    @property
    def T(self): # Attribute 4
        if self.ndim != 2: 
            raise ValueError("T only supports 2D arrays in this lightweight version")
        r, c = self._shape
        new_data = [0.0] * self.size
        for i in range(r):
            for j in range(c):
                new_data[j * r + i] = self.data[i * c + j]
        return ndarray(new_data, (c, r))

    def tolist(self): # Function 5 (Method)
        flat_list = []
        for val in self.data:
            flat_list.append(val)
        idx = [0]
        def reconstruct(shp):
            if not shp:
                val = flat_list[idx[0]]
                idx[0] += 1
                return val
            res = []
            for _ in range(shp[0]):
                res.append(reconstruct(shp[1:]))
            return res
        return reconstruct(self._shape)

    def __repr__(self):
        return f"array({self.tolist()})"


# --- 2. MATHEMATICAL CONSTANTS ---
pi = 3.141592653589793 # Constant 6
e = 2.718281828459045  # Constant 7


# --- 3. LOW-LEVEL ALGORITHM UTILITIES (No Built-ins Allowed) ---

def _sqrt(x):
    if x < 0: raise ValueError("Math domain error")
    if x == 0: return 0.0
    g = x / 2.0
    for _ in range(15): 
        g = 0.5 * (g + x / g)
    return g

def _sin(x):
    x = x % (2 * pi)
    term = x
    s = x
    sign = -1
    for i in range(3, 25, 2):
        term = term * x * x / (i * (i - 1))
        s += sign * term
        sign *= -1
    return s

def _cos(x): 
    return _sin(x + pi / 2)

def _exp(x):
    s = 1.0
    term = 1.0
    for i in range(1, 30):
        term *= x / i
        s += term
    return s

def _log(x):
    if x <= 0: raise ValueError("Math domain error")
    n = 0
    while x > 2.0: 
        x /= e
        n += 1
    while x < 1.0: 
        x *= e
        n -= 1
    z = (x - 1) / (x + 1)
    s = z
    term = z
    z2 = z * z
    for i in range(3, 60, 2):
        term *= z2
        s += term / i
    return 2 * s + n

def _sort_list(lst):
    res = []
    for x in lst:
        res.append(x)
    n = len(res)
    for i in range(n):
        for j in range(0, n - i - 1):
            if res[j] > res[j + 1]:
                res[j], res[j + 1] = res[j + 1], res[j]
    return res


# --- 4. THE 50 FUNCTION CORE IMPLEMENTATIONS ---

def array(data): return ndarray(data) # Function 8

def zeros(shape): # Function 9
    p = 1
    for x in shape: p *= x
    return ndarray([0.0] * p, shape)

def ones(shape): # Function 10
    p = 1
    for x in shape: p *= x
    return ndarray([1.0] * p, shape)

def arange(*args): # Function 11
    start, step = 0, 1
    if len(args) == 1: end = args[0]
    elif len(args) == 2: start, end = args[0], args[1]
    else: start, end, step = args[0], args[1], args[2]
    res = []
    val = start
    while (step > 0 and val < end) or (step < 0 and val > end):
        res.append(val)
        val += step
    return ndarray(res)

def linspace(start, end, num=50): # Function 12
    if num == 1: return array([start])
    step = (end - start) / (num - 1)
    res = []
    for i in range(num):
        res.append(start + i * step)
    return array(res)

def eye(n): # Function 13
    data = []
    for i in range(n):
        for j in range(n):
            if i == j: data.append(1.0)
            else: data.append(0.0)
    return ndarray(data, (n, n))

def reshape(a, new_shape): # Function 14
    p = 1
    for x in new_shape: p *= x
    if p != a.size: raise ValueError("Total size of new array must be unchanged")
    return ndarray(a.data, new_shape)

def ravel(a): return ndarray(a.data, (a.size,)) # Function 15

def concatenate(arrays): # Function 16
    combined = []
    for arr in arrays: 
        for val in arr.data:
            combined.append(val)
    return ndarray(combined)

def transpose(a): return a.T # Function 17

def sum(a): # Function 18
    total = 0.0
    for val in a.data: 
        total += val
    return total

def mean(a): return sum(a) / a.size # Function 19

def min(a): # Function 20
    if not a.data: raise ValueError("Empty array")
    lowest = a.data[0]
    for val in a.data:
        if val < lowest: lowest = val
    return lowest

def max(a): # Function 21
    if not a.data: raise ValueError("Empty array")
    highest = a.data[0]
    for val in a.data:
        if val > highest: highest = val
    return highest

def argmin(a): # Function 22
    if not a.data: raise ValueError("Empty array")
    lowest = a.data[0]
    low_idx = 0
    for idx, val in enumerate(a.data):
        if val < lowest:
            lowest = val
            low_idx = idx
    return low_idx

def argmax(a): # Function 23
    if not a.data: raise ValueError("Empty array")
    highest = a.data[0]
    high_idx = 0
    for idx, val in enumerate(a.data):
        if val > highest:
            highest = val
            high_idx = idx
    return high_idx

def sqrt(a): # Function 24
    res = []
    for x in a.data: res.append(_sqrt(x))
    return ndarray(res, a.shape)

def sin(a): # Function 25
    res = []
    for x in a.data: res.append(_sin(x))
    return ndarray(res, a.shape)

def cos(a): # Function 26
    res = []
    for x in a.data: res.append(_cos(x))
    return ndarray(res, a.shape)

def exp(a): # Function 27
    res = []
    for x in a.data: res.append(_exp(x))
    return ndarray(res, a.shape)

def log(a): # Function 28
    res = []
    for x in a.data: res.append(_log(x))
    return ndarray(res, a.shape)

def abs(a): # Function 29
    res = []
    for x in a.data:
        if x < 0: res.append(-x)
        else: res.append(x)
    return ndarray(res, a.shape)

def round(a, decimals=0): # Function 30
    factor = 10 ** decimals
    res = []
    for x in a.data:
        shifted = x * factor
        integer_part = int(shifted)
        fraction = shifted - integer_part
        if fraction >= 0.5: integer_part += 1
        elif fraction <= -0.5: integer_part -= 1
        res.append(integer_part / factor)
    return ndarray(res, a.shape)

def floor(a): # Function 31
    res = []
    for x in a.data:
        if x >= 0: res.append(float(int(x)))
        else: res.append(float(int(x) - 1))
    return ndarray(res, a.shape)

def ceil(a): # Function 32
    res = []
    for x in a.data:
        if x <= 0: res.append(float(int(x)))
        else:
            if x == int(x): res.append(float(int(x)))
            else: res.append(float(int(x) + 1))
    return ndarray(res, a.shape)

def add(a, b): # Function 33
    res = []
    for i in range(a.size): res.append(a.data[i] + b.data[i])
    return ndarray(res, a.shape)

def subtract(a, b): # Function 34
    res = []
    for i in range(a.size): res.append(a.data[i] - b.data[i])
    return ndarray(res, a.shape)

def multiply(a, b): # Function 35
    res = []
    for i in range(a.size): res.append(a.data[i] * b.data[i])
    return ndarray(res, a.shape)

def divide(a, b): # Function 36
    res = []
    for i in range(a.size): res.append(a.data[i] / b.data[i])
    return ndarray(res, a.shape)

def power(a, b): # Function 37
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        exponent = b.data[i] if is_arr else b
        res.append(a.data[i] ** exponent)
    return ndarray(res, a.shape)

def dot(a, b): # Function 38
    if a.ndim != 2 or b.ndim != 2: raise ValueError("Only 2D matrices supported here")
    r1, c1 = a.shape
    r2, c2 = b.shape
    if c1 != r2: raise ValueError("Matrix dimension mismatch")
    res = zeros((r1, c2))
    for i in range(r1):
        for j in range(c2):
            s = 0.0
            for k in range(c1):
                s += a.data[i * c1 + k] * b.data[k * c2 + j]
            res.data[i * c2 + j] = s
    return res

def clip(a, a_min, a_max): # Function 39
    res = []
    for x in a.data:
        if x < a_min: res.append(a_min)
        elif x > a_max: res.append(a_max)
        else: res.append(x)
    return ndarray(res, a.shape)

def unique(a): # Function 40
    uniq = []
    for x in a.data:
        if x not in uniq: uniq.append(x)
    return ndarray(_sort_list(uniq))

def sort(a): # Function 41
    return ndarray(_sort_list(a.data), a.shape)

def argsort(a): # Function 42
    indices = []
    for i in range(len(a.data)): indices.append(i)
    n = len(indices)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a.data[indices[j]] > a.data[indices[j + 1]]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
    return ndarray(indices, a.shape)

def std(a): # Function 43
    m = mean(a)
    variance_sum = 0.0
    for x in a.data:
        variance_sum += (x - m) ** 2
    return _sqrt(variance_sum / a.size)

def var(a): return std(a) ** 2 # Function 44

  # --- 5. EXTENDED 100-FUNCTION NUMPY-LIKE MATRIX & ARRAY API ---

# --- ADDITIONAL LOW-LEVEL MATH UTILITIES ---

def _atan(x):
    """Taylor series approximation for arctan(x)."""
    if x > 1.0:
        return (pi / 2) - _atan(1.0 / x)
    if x < -1.0:
        return (-pi / 2) - _atan(1.0 / x)
    s, term, x2 = x, x, x * x
    for i in range(3, 40, 2):
        term *= -x2
        s += term / i
    return s

def _atan2(y, x):
    if x > 0: return _atan(y / x)
    if x < 0 and y >= 0: return _atan(y / x) + pi
    if x < 0 and y < 0: return _atan(y / x) - pi
    if x == 0 and y > 0: return pi / 2
    if x == 0 and y < 0: return -pi / 2
    return 0.0

def _tan(x):
    c = _cos(x)
    if c == 0: raise ZeroDivisionError("Math domain error: tan undefined")
    return _sin(x) / c


# --- MATHEMATICAL CONSTANTS ---
tau = 2 * pi          # Constant 45
euler_gamma = 0.5772156649015328 # Constant 46


# --- ADDITIONAL ARRAY CREATION & SHAPING ---

def full(shape, fill_value): # Function 47
    p = 1
    for x in shape: p *= x
    return ndarray([fill_value] * p, shape)

def full_like(a, fill_value): # Function 48
    return full(a.shape, fill_value)

def zeros_like(a): # Function 49
    return zeros(a.shape)

def ones_like(a): # Function 50
    return ones(a.shape)

def identity(n): # Function 51
    return eye(n)

def diag(v, k=0): # Function 52
    if v.ndim == 1:
        n = v.size + abs(k)
        res = zeros((n, n))
        for i in range(v.size):
            r = i if k >= 0 else i - k
            c = i + k if k >= 0 else i
            res.data[r * n + c] = v.data[i]
        return res
    elif v.ndim == 2:
        r, c = v.shape
        diag_vals = []
        for i in range(min(r, c)):
            diag_vals.append(v.data[i * c + i])
        return ndarray(diag_vals)
    raise ValueError("Input must be 1D or 2D array")

def triu(m, k=0): # Function 53
    if m.ndim != 2: raise ValueError("Matrix must be 2D")
    r, c = m.shape
    res = zeros((r, c))
    for i in range(r):
        for j in range(c):
            if j >= i + k:
                res.data[i * c + j] = m.data[i * c + j]
    return res

def tril(m, k=0): # Function 54
    if m.ndim != 2: raise ValueError("Matrix must be 2D")
    r, c = m.shape
    res = zeros((r, c))
    for i in range(r):
        for j in range(c):
            if j <= i + k:
                res.data[i * c + j] = m.data[i * c + j]
    return res

def squeeze(a): # Function 55
    new_shape = []
    for x in a.shape:
        if x != 1: new_shape.append(x)
    if not new_shape: new_shape = (1,)
    return ndarray(a.data, tuple(new_shape))

def expand_dims(a, axis): # Function 56
    shp = list(a.shape)
    if axis < 0: axis += len(shp) + 1
    shp.insert(axis, 1)
    return ndarray(a.data, tuple(shp))

def flatten(a): # Function 57
    return ravel(a)

def repeat(a, repeats): # Function 58
    res = []
    for val in a.data:
        for _ in range(repeats):
            res.append(val)
    return ndarray(res)

def tile(a, reps): # Function 59
    res = []
    for _ in range(reps):
        for val in a.data:
            res.append(val)
    return ndarray(res)


# --- ADDITIONAL TRIGONOMETRIC & ELEMENT-WISE MATH ---

def tan(a): # Function 60
    res = []
    for x in a.data: res.append(_tan(x))
    return ndarray(res, a.shape)

def arcsin(a): # Function 61
    res = []
    for x in a.data:
        if x < -1.0 or x > 1.0: raise ValueError("Math domain error")
        res.append(_atan2(x, _sqrt(1.0 - x * x)))
    return ndarray(res, a.shape)

def arccos(a): # Function 62
    res = []
    for x in a.data:
        if x < -1.0 or x > 1.0: raise ValueError("Math domain error")
        res.append(_atan2(_sqrt(1.0 - x * x), x))
    return ndarray(res, a.shape)

def arctan(a): # Function 63
    res = []
    for x in a.data: res.append(_atan(x))
    return ndarray(res, a.shape)

def arctan2(a, b): # Function 64
    res = []
    for i in range(a.size):
        res.append(_atan2(a.data[i], b.data[i]))
    return ndarray(res, a.shape)

def sinh(a): # Function 65
    res = []
    for x in a.data: res.append((_exp(x) - _exp(-x)) / 2.0)
    return ndarray(res, a.shape)

def cosh(a): # Function 66
    res = []
    for x in a.data: res.append((_exp(x) + _exp(-x)) / 2.0)
    return ndarray(res, a.shape)

def tanh(a): # Function 67
    res = []
    for x in a.data:
        ex = _exp(x)
        enx = _exp(-x)
        res.append((ex - enx) / (ex + enx))
    return ndarray(res, a.shape)

def rad2deg(a): # Function 68
    res = []
    for x in a.data: res.append(x * 180.0 / pi)
    return ndarray(res, a.shape)

def deg2rad(a): # Function 69
    res = []
    for x in a.data: res.append(x * pi / 180.0)
    return ndarray(res, a.shape)

def log2(a): # Function 70
    res = []
    ln2 = _log(2.0)
    for x in a.data: res.append(_log(x) / ln2)
    return ndarray(res, a.shape)

def log10(a): # Function 71
    res = []
    ln10 = _log(10.0)
    for x in a.data: res.append(_log(x) / ln10)
    return ndarray(res, a.shape)

def square(a): # Function 72
    res = []
    for x in a.data: res.append(x * x)
    return ndarray(res, a.shape)

def reciprocal(a): # Function 73
    res = []
    for x in a.data: res.append(1.0 / x)
    return ndarray(res, a.shape)

def sign(a): # Function 74
    res = []
    for x in a.data:
        if x > 0: res.append(1.0)
        elif x < 0: res.append(-1.0)
        else: res.append(0.0)
    return ndarray(res, a.shape)

def fmod(a, b): # Function 75
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        mod = b.data[i] if is_arr else b
        res.append(a.data[i] - int(a.data[i] / mod) * mod)
    return ndarray(res, a.shape)


# --- COMPARISONS & LOGICAL UTILITIES ---

def equal(a, b): # Function 76
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if a.data[i] == other else 0.0)
    return ndarray(res, a.shape)

def greater(a, b): # Function 77
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if a.data[i] > other else 0.0)
    return ndarray(res, a.shape)

def less(a, b): # Function 78
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if a.data[i] < other else 0.0)
    return ndarray(res, a.shape)

def isclose(a, b, rtol=1e-05, atol=1e-08): # Function 79
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        target = b.data[i] if is_arr else b
        diff = abs(a.data[i] - target)
        bound = atol + rtol * abs(target)
        res.append(1.0 if diff <= bound else 0.0)
    return ndarray(res, a.shape)

def all(a): # Function 80
    for x in a.data:
        if x == 0: return False
    return True

def any(a): # Function 81
    for x in a.data:
        if x != 0: return True
    return False

def where(condition, x, y): # Function 82
    res = []
    is_x_arr = isinstance(x, ndarray)
    is_y_arr = isinstance(y, ndarray)
    for i in range(condition.size):
        if condition.data[i] != 0:
            res.append(x.data[i] if is_x_arr else x)
        else:
            res.append(y.data[i] if is_y_arr else y)
    return ndarray(res, condition.shape)


# --- STATISTICAL & REDUCTION UTILITIES ---

def prod(a): # Function 83
    p = 1.0
    for x in a.data: p *= x
    return p

def median(a): # Function 84
    sorted_a = _sort_list(a.data)
    n = len(sorted_a)
    if n == 0: raise ValueError("Empty array")
    if n % 2 == 1:
        return float(sorted_a[n // 2])
    else:
        return (sorted_a[n // 2 - 1] + sorted_a[n // 2]) / 2.0

def ptp(a): # Function 85
    return max(a) - min(a)

def cumsum(a): # Function 86
    res = []
    total = 0.0
    for x in a.data:
        total += x
        res.append(total)
    return ndarray(res, a.shape)

def cumprod(a): # Function 87
    res = []
    total = 1.0
    for x in a.data:
        total *= x
        res.append(total)
    return ndarray(res, a.shape)

def diff(a): # Function 88
    if a.size <= 1: return ndarray([])
    res = []
    for i in range(a.size - 1):
        res.append(a.data[i + 1] - a.data[i])
    return ndarray(res)

def trapz(y, x=None): # Function 89
    n = y.size
    if n <= 1: return 0.0
    total = 0.0
    if x is None:
        for i in range(n - 1):
            total += 0.5 * (y.data[i] + y.data[i + 1])
    else:
        for i in range(n - 1):
            dx = x.data[i + 1] - x.data[i]
            total += 0.5 * (y.data[i] + y.data[i + 1]) * dx
    return total


# --- LINEAR ALGEBRA & VECTOR UTILITIES ---

def norm(a): # Function 90
    total = 0.0
    for x in a.data: total += x * x
    return _sqrt(total)

def trace(a): # Function 91
    if a.ndim != 2: raise ValueError("Matrix must be 2D")
    r, c = a.shape
    total = 0.0
    for i in range(min(r, c)):
        total += a.data[i * c + i]
    return total

def outer(a, b): # Function 92
    res = []
    for i in range(a.size):
        for j in range(b.size):
            res.append(a.data[i] * b.data[j])
    return ndarray(res, (a.size, b.size))

def inner(a, b): # Function 93
    if a.size != b.size: raise ValueError("Arrays must be equal size")
    s = 0.0
    for i in range(a.size):
        s += a.data[i] * b.data[i]
    return s

def cross(a, b): # Function 94
    if a.size != 3 or b.size != 3:
        raise ValueError("Both vectors must be 3-dimensional")
    u, v = a.data, b.data
    res = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    ]
    return ndarray(res)

def vdot(a, b): # Function 95
    return inner(a, b)

def det2x2(a): # Function 96
    if a.shape != (2, 2): raise ValueError("Array must be a 2x2 matrix")
    d = a.data
    return d[0] * d[3] - d[1] * d[2]

def inv2x2(a): # Function 97
    d = det2x2(a)
    if d == 0: raise ValueError("Singular matrix cannot be inverted")
    m = a.data
    inv_data = [
        m[3] / d, -m[1] / d,
        -m[2] / d, m[0] / d
    ]
    return ndarray(inv_data, (2, 2))

def kron(a, b): # Function 98
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Kronecker product here supports 2D matrices")
    ra, ca = a.shape
    rb, cb = b.shape
    res_r, res_c = ra * rb, ca * cb
    res = zeros((res_r, res_c))
    for i in range(ra):
        for j in range(ca):
            val_a = a.data[i * ca + j]
            for k in range(rb):
                for l in range(cb):
                    val_b = b.data[k * cb + l]
                    res.data[(i * rb + k) * res_c + (j * cb + l)] = val_a * val_b
    return res

def cov(a, b): # Function 99
    if a.size != b.size: raise ValueError("Arrays must be equal size")
    ma, mb = mean(a), mean(b)
    s = 0.0
    for i in range(a.size):
        s += (a.data[i] - ma) * (b.data[i] - mb)
    return s / (a.size - 1)

def corrcoef(a, b): # Function 100
    covariance = cov(a, b)
    std_a, std_b = std(a), std(b)
    if std_a == 0 or std_b == 0: return 0.0
    return covariance / (std_a * std_b)

  # --- 101 to 180: EXTENDED NUMPY DEFINITIONS ---


# --- ARRAY MANIPULATION & SLICING ---


def split(a, indices_or_sections):  # Function 101
    n = a.size
    if isinstance(indices_or_sections, int):
        sec = indices_or_sections
        if n % sec != 0:
            raise ValueError("array split does not result in an equal division")
        step = n // sec
        splits = []
        for i in range(sec):
            splits.append(ndarray(a.data[i * step : (i + 1) * step]))
        return splits
    else:
        splits, last = [], 0
        for idx in indices_or_sections:
            splits.append(ndarray(a.data[last:idx]))
            last = idx
        splits.append(ndarray(a.data[last:]))
        return splits


def array_split(a, indices_or_sections):  # Function 102
    n = a.size
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        nelem = n // sections
        ext = n % sections
        splits, idx = [], 0
        for i in range(sections):
            length = nelem + (1 if i < ext else 0)
            splits.append(ndarray(a.data[idx : idx + length]))
            idx += length
        return splits
    return split(a, indices_or_sections)


def hstack(tup):  # Function 103
    return concatenate(tup)


def vstack(tup):  # Function 104
    if not tup:
        raise ValueError("need at least one array")
    if tup[0].ndim == 1:
        combined = []
        for arr in tup:
            combined.extend(arr.data)
        return ndarray(combined, (len(tup), tup[0].size))
    r_total = sum(arr.shape[0] for arr in tup)
    c = tup[0].shape[1]
    combined = []
    for arr in tup:
        combined.extend(arr.data)
    return ndarray(combined, (r_total, c))


def dstack(tup):  # Function 105
    combined = []
    length = tup[0].size
    for i in range(length):
        for arr in tup:
            combined.append(arr.data[i])
    return ndarray(combined)


def column_stack(tup):  # Function 106
    return vstack(tup).T


def row_stack(tup):  # Function 107
    return vstack(tup)


def flip(a):  # Function 108
    res = []
    for i in range(a.size - 1, -1, -1):
        res.append(a.data[i])
    return ndarray(res, a.shape)


def fliplr(a):  # Function 109
    if a.ndim != 2:
        raise ValueError("Input must be >= 2-d.")
    r, c = a.shape
    res = [0.0] * a.size
    for i in range(r):
        for j in range(c):
            res[i * c + (c - 1 - j)] = a.data[i * c + j]
    return ndarray(res, a.shape)


def flipud(a):  # Function 110
    if a.ndim != 2:
        raise ValueError("Input must be >= 2-d.")
    r, c = a.shape
    res = [0.0] * a.size
    for i in range(r):
        for j in range(c):
            res[(r - 1 - i) * c + j] = a.data[i * c + j]
    return ndarray(res, a.shape)


def roll(a, shift):  # Function 111
    n = a.size
    shift = shift % n
    res = [0.0] * n
    for i in range(n):
        res[(i + shift) % n] = a.data[i]
    return ndarray(res, a.shape)


def rot90(a, k=1):  # Function 112
    if a.ndim != 2:
        raise ValueError("Input must be 2D")
    k = k % 4
    res = a
    for _ in range(k):
        r, c = res.shape
        new_data = [0.0] * res.size
        for i in range(r):
            for j in range(c):
                new_data[(c - 1 - j) * r + i] = res.data[i * c + j]
        res = ndarray(new_data, (c, r))
    return res


def trim_zeros(a, trim="fb"):  # Function 113
    start, end = 0, a.size
    if "f" in trim.lower():
        while start < end and a.data[start] == 0:
            start += 1
    if "b" in trim.lower():
        while end > start and a.data[end - 1] == 0:
            end -= 1
    return ndarray(a.data[start:end])


def extract(condition, a):  # Function 114
    res = []
    for i in range(min(condition.size, a.size)):
        if condition.data[i] != 0:
            res.append(a.data[i])
    return ndarray(res)


def place(a, mask, vals):  # Function 115
    res = list(a.data)
    val_idx = 0
    is_arr = isinstance(vals, ndarray)
    val_list = vals.data if is_arr else vals
    for i in range(min(a.size, mask.size)):
        if mask.data[i] != 0:
            res[i] = (
                val_list[val_idx % len(val_list)]
                if isinstance(val_list, list)
                else val_list
            )
            val_idx += 1
    return ndarray(res, a.shape)


def put(a, ind, v):  # Function 116
    res = list(a.data)
    is_v_arr = isinstance(v, ndarray)
    v_list = v.data if is_v_arr else (v if isinstance(v, list) else [v])
    ind_list = ind.data if isinstance(ind, ndarray) else ind
    for i, idx in enumerate(ind_list):
        res[idx] = v_list[i % len(v_list)]
    return ndarray(res, a.shape)


def take(a, indices):  # Function 117
    res = []
    ind_list = indices.data if isinstance(indices, ndarray) else indices
    for idx in ind_list:
        res.append(a.data[idx])
    return ndarray(
        res, indices.shape if isinstance(indices, ndarray) else (len(indices),)
    )


def compress(condition, a):  # Function 118
    return extract(condition, a)


# --- SPECIAL CREATORS & GRID GENERATION ---


def logspace(start, stop, num=50, base=10.0):  # Function 119
    lin = linspace(start, stop, num)
    res = []
    for x in lin.data:
        res.append(base**x)
    return ndarray(res)


def geomspace(start, stop, num=50):  # Function 120
    if start == 0 or stop == 0:
        raise ValueError("Geometric sequence cannot include zero")
    log_start = _log(abs(start)) / _log(10.0)
    log_stop = _log(abs(stop)) / _log(10.0)
    ls = logspace(log_start, log_stop, num)
    if start < 0:
        return multiply(ls, array([-1.0]))
    return ls


def meshgrid(x, y):  # Function 121
    nx, ny = x.size, y.size
    X_data, Y_data = [], []
    for j in range(ny):
        for i in range(nx):
            X_data.append(x.data[i])
            Y_data.append(y.data[j])
    return ndarray(X_data, (ny, nx)), ndarray(Y_data, (ny, nx))


def indices(dimensions):  # Function 122
    if len(dimensions) != 2:
        raise ValueError("Only 2D dimensions supported")
    r, c = dimensions
    grid_r, grid_c = [], []
    for i in range(r):
        for j in range(c):
            grid_r.append(float(i))
            grid_c.append(float(j))
    return ndarray(grid_r, (r, c)), ndarray(grid_c, (r, c))


def pad(a, pad_width, constant_values=0.0):  # Function 123
    if a.ndim != 1:
        raise ValueError("Pad currently supports 1D arrays")
    pw = pad_width if isinstance(pad_width, tuple) else (pad_width, pad_width)
    res = [constant_values] * pw[0] + list(a.data) + [constant_values] * pw[1]
    return ndarray(res)


# --- MATH, BITWISE & NUMERICAL EXPANSIONS ---


def cbrt(a):  # Function 124
    res = []
    for x in a.data:
        if x >= 0:
            res.append(x ** (1.0 / 3.0))
        else:
            res.append(-((-x) ** (1.0 / 3.0)))
    return ndarray(res, a.shape)


def hypot(a, b):  # Function 125
    res = []
    for i in range(a.size):
        res.append(_sqrt(a.data[i] ** 2 + b.data[i] ** 2))
    return ndarray(res, a.shape)


def sinc(a):  # Function 126
    res = []
    for x in a.data:
        px = pi * x
        if px == 0:
            res.append(1.0)
        else:
            res.append(_sin(px) / px)
    return ndarray(res, a.shape)


def expm1(a):  # Function 127
    res = []
    for x in a.data:
        res.append(_exp(x) - 1.0)
    return ndarray(res, a.shape)


def log1p(a):  # Function 128
    res = []
    for x in a.data:
        res.append(_log(1.0 + x))
    return ndarray(res, a.shape)


def copysign(a, b):  # Function 129
    res = []
    is_b_arr = isinstance(b, ndarray)
    for i in range(a.size):
        s = b.data[i] if is_b_arr else b
        val = abs(a.data[i])
        res.append(val if s >= 0 else -val)
    return ndarray(res, a.shape)


def frexp(a):  # Function 130
    mantissas, exponents = [], []
    for x in a.data:
        if x == 0:
            mantissas.append(0.0)
            exponents.append(0)
        else:
            exp_val = int(_log(abs(x)) / _log(2.0)) + 1
            mant = x / (2.0**exp_val)
            mantissas.append(mant)
            exponents.append(exp_val)
    return ndarray(mantissas, a.shape), ndarray(exponents, a.shape)


def ldexp(x1, x2):  # Function 131
    res = []
    is_arr = isinstance(x2, ndarray)
    for i in range(x1.size):
        e = x2.data[i] if is_arr else x2
        res.append(x1.data[i] * (2.0**e))
    return ndarray(res, x1.shape)


def gcd(a, b):  # Function 132
    res = []
    is_b_arr = isinstance(b, ndarray)
    for i in range(a.size):
        x = int(abs(a.data[i]))
        y = int(abs(b.data[i] if is_b_arr else b))
        while y:
            x, y = y, x % y
        res.append(float(x))
    return ndarray(res, a.shape)


def lcm(a, b):  # Function 133
    res = []
    is_b_arr = isinstance(b, ndarray)
    for i in range(a.size):
        x = int(abs(a.data[i]))
        y = int(abs(b.data[i] if is_b_arr else b))
        if x == 0 or y == 0:
            res.append(0.0)
        else:
            orig_x, orig_y = x, y
            while y:
                x, y = y, x % y
            res.append(float((orig_x * orig_y) // x))
    return ndarray(res, a.shape)


def modf(a):  # Function 134
    fractional, integral = [], []
    for x in a.data:
        int_p = float(int(x))
        fractional.append(x - int_p)
        integral.append(int_p)
    return ndarray(fractional, a.shape), ndarray(integral, a.shape)


def bitwise_and(a, b):  # Function 135
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = int(b.data[i] if is_arr else b)
        res.append(float(int(a.data[i]) & other))
    return ndarray(res, a.shape)


def bitwise_or(a, b):  # Function 136
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = int(b.data[i] if is_arr else b)
        res.append(float(int(a.data[i]) | other))
    return ndarray(res, a.shape)


def bitwise_xor(a, b):  # Function 137
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = int(b.data[i] if is_arr else b)
        res.append(float(int(a.data[i]) ^ other))
    return ndarray(res, a.shape)


def invert(a):  # Function 138
    res = []
    for x in a.data:
        res.append(float(~int(x)))
    return ndarray(res, a.shape)


def left_shift(a, b):  # Function 139
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        s = int(b.data[i] if is_arr else b)
        res.append(float(int(a.data[i]) << s))
    return ndarray(res, a.shape)


def right_shift(a, b):  # Function 140
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        s = int(b.data[i] if is_arr else b)
        res.append(float(int(a.data[i]) >> s))
    return ndarray(res, a.shape)


# --- LOGICAL OPERATORS & TESTING ---


def logical_and(a, b):  # Function 141
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if (a.data[i] != 0 and other != 0) else 0.0)
    return ndarray(res, a.shape)


def logical_or(a, b):  # Function 142
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if (a.data[i] != 0 or other != 0) else 0.0)
    return ndarray(res, a.shape)


def logical_not(a):  # Function 143
    res = []
    for x in a.data:
        res.append(1.0 if x == 0 else 0.0)
    return ndarray(res, a.shape)


def logical_xor(a, b):  # Function 144
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        cond1 = a.data[i] != 0
        cond2 = other != 0
        res.append(1.0 if (cond1 != cond2) else 0.0)
    return ndarray(res, a.shape)


def greater_equal(a, b):  # Function 145
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if a.data[i] >= other else 0.0)
    return ndarray(res, a.shape)


def less_equal(a, b):  # Function 146
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if a.data[i] <= other else 0.0)
    return ndarray(res, a.shape)


def not_equal(a, b):  # Function 147
    res = []
    is_arr = isinstance(b, ndarray)
    for i in range(a.size):
        other = b.data[i] if is_arr else b
        res.append(1.0 if a.data[i] != other else 0.0)
    return ndarray(res, a.shape)


def isfinite(a):  # Function 148
    res = []
    for x in a.data:
        res.append(1.0 if (x == x and x != float("inf") and x != float("-inf")) else 0.0)
    return ndarray(res, a.shape)


def isinf(a):  # Function 149
    res = []
    for x in a.data:
        res.append(1.0 if (x == float("inf") or x == float("-inf")) else 0.0)
    return ndarray(res, a.shape)


def isnan(a):  # Function 150
    res = []
    for x in a.data:
        res.append(1.0 if x != x else 0.0)
    return ndarray(res, a.shape)


def count_nonzero(a):  # Function 151
    count = 0
    for x in a.data:
        if x != 0:
            count += 1
    return count


def flatnonzero(a):  # Function 152
    res = []
    for i, x in enumerate(a.data):
        if x != 0:
            res.append(i)
    return ndarray(res)


def argwhere(a):  # Function 153
    res = []
    for i, x in enumerate(a.data):
        if x != 0:
            res.append(i)
    return ndarray(res)


def nonzero(a):  # Function 154
    return (flatnonzero(a),)


# --- ADVANCED SEARCH & SET UTILITIES ---


def searchsorted(a, v):  # Function 155
    res = []
    v_list = v.data if isinstance(v, ndarray) else [v]
    for val in v_list:
        idx = 0
        while idx < a.size and a.data[idx] < val:
            idx += 1
        res.append(idx)
    return ndarray(res) if isinstance(v, ndarray) else res[0]


def isin(element, test_elements):  # Function 156
    test_set = (
        test_elements.data
        if isinstance(test_elements, ndarray)
        else test_elements
    )
    res = []
    for x in element.data:
        res.append(1.0 if x in test_set else 0.0)
    return ndarray(res, element.shape)


def setdiff1d(ar1, ar2):  # Function 157
    s2 = ar2.data if isinstance(ar2, ndarray) else ar2
    res = []
    for x in ar1.data:
        if x not in s2 and x not in res:
            res.append(x)
    return ndarray(_sort_list(res))


def intersect1d(ar1, ar2):  # Function 158
    s2 = ar2.data if isinstance(ar2, ndarray) else ar2
    res = []
    for x in ar1.data:
        if x in s2 and x not in res:
            res.append(x)
    return ndarray(_sort_list(res))


def union1d(ar1, ar2):  # Function 159
    res = []
    for x in ar1.data:
        if x not in res:
            res.append(x)
    for x in ar2.data:
        if x not in res:
            res.append(x)
    return ndarray(_sort_list(res))


def setxor1d(ar1, ar2):  # Function 160
    s1, s2 = ar1.data, ar2.data
    res = []
    for x in s1:
        if x not in s2 and x not in res:
            res.append(x)
    for x in s2:
        if x not in s1 and x not in res:
            res.append(x)
    return ndarray(_sort_list(res))


# --- ADVANCED AGGREGATIONS & STATISTICS ---


def nanmean(a):  # Function 161
    total, count = 0.0, 0
    for x in a.data:
        if x == x:
            total += x
            count += 1
    return total / count if count > 0 else float("nan")


def nansum(a):  # Function 162
    total = 0.0
    for x in a.data:
        if x == x:
            total += x
    return total


def nanstd(a):  # Function 163
    m = nanmean(a)
    total, count = 0.0, 0
    for x in a.data:
        if x == x:
            total += (x - m) ** 2
            count += 1
    return _sqrt(total / count) if count > 0 else float("nan")


def nanvar(a):  # Function 164
    s = nanstd(a)
    return s * s


def nanmin(a):  # Function 165
    valid = [x for x in a.data if x == x]
    return min(ndarray(valid))


def nanmax(a):  # Function 166
    valid = [x for x in a.data if x == x]
    return max(ndarray(valid))


def percentile(a, q):  # Function 167
    s = _sort_list(a.data)
    n = len(s)
    if n == 0:
        raise ValueError("Empty array")
    k = (n - 1) * (q / 100.0)
    f = int(k)
    c = f + 1
    if c < n:
        return s[f] + (k - f) * (s[c] - s[f])
    return float(s[f])


def quantile(a, q):  # Function 168
    return percentile(a, q * 100.0)


def average(a, weights=None):  # Function 169
    if weights is None:
        return mean(a)
    w_sum = sum(weights)
    if w_sum == 0:
        raise ZeroDivisionError("Weights sum to zero")
    dot_p = inner(a, weights)
    return dot_p / w_sum


def bincount(x):  # Function 170
    if not x.data:
        return ndarray([])
    m = int(max(x))
    counts = [0] * (m + 1)
    for val in x.data:
        counts[int(val)] += 1
    return ndarray(counts)


def histogram(a, bins=10):  # Function 171
    mn, mx = min(a), max(a)
    if mn == mx:
        mx += 1.0
    step = (mx - mn) / bins
    counts = [0] * bins
    bin_edges = [mn + i * step for i in range(bins + 1)]
    for val in a.data:
        if val == mx:
            counts[-1] += 1
        else:
            idx = int((val - mn) / step)
            if 0 <= idx < bins:
                counts[idx] += 1
    return ndarray(counts), ndarray(bin_edges)


# --- LINEAR ALGEBRA EXTENSIONS & FINANCIALS ---


def eye_like(a):  # Function 172
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Requires 2D square matrix")
    return eye(a.shape[0])


def matrix_power(a, n):  # Function 173
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix")
    if n == 0:
        return eye(a.shape[0])
    res = a
    for _ in range(n - 1):
        res = dot(res, a)
    return res


def cond2x2(a):  # Function 174
    det = det2x2(a)
    if det == 0:
        return float("inf")
    n_a = norm(a)
    inv_a = inv2x2(a)
    n_inv = norm(inv_a)
    return n_a * n_inv


def polyval(p, x):  # Function 175
    res = []
    x_list = x.data if isinstance(x, ndarray) else [x]
    for val in x_list:
        y = 0.0
        for coeff in p.data:
            y = y * val + coeff
        res.append(y)
    return ndarray(res, x.shape) if isinstance(x, ndarray) else res[0]


def polyder(p):  # Function 176
    n = p.size
    if n <= 1:
        return ndarray([0.0])
    res = []
    for i in range(n - 1):
        power_val = n - 1 - i
        res.append(p.data[i] * power_val)
    return ndarray(res)


def polyint(p, c=0.0):  # Function 177
    n = p.size
    res = []
    for i in range(n):
        power_val = n - i
        res.append(p.data[i] / power_val)
    res.append(c)
    return ndarray(res)


def npv(rate, values):  # Function 178
    total = 0.0
    for i, val in enumerate(values.data):
        total += val / ((1.0 + rate) ** i)
    return total


def irr(values):  # Function 179
    rate = 0.1
    for _ in range(100):
        val = npv(rate, values)
        # Numerical derivative
        d_val = (npv(rate + 1e-5, values) - val) / 1e-5
        if abs(d_val) < 1e-12:
            break
        rate -= val / d_val
    return rate


def interp(x, xp, fp):  # Function 180
    res = []
    x_list = x.data if isinstance(x, ndarray) else [x]
    for val in x_list:
        if val <= xp.data[0]:
            res.append(fp.data[0])
        elif val >= xp.data[-1]:
            res.append(fp.data[-1])
        else:
            i = 0
            while i < xp.size - 1 and xp.data[i + 1] < val:
                i += 1
            slope = (fp.data[i + 1] - fp.data[i]) / (xp.data[i + 1] - xp.data[i])
            res.append(fp.data[i] + slope * (val - xp.data[i]))
    return ndarray(res, x.shape) if isinstance(x, ndarray) else res[0]



