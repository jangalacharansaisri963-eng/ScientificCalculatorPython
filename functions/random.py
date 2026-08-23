# ==============================================================================
# PURE PYTHON INDEPENDENT RANDOM MODULE (NO IMPORTS)
# Expanded Architecture & Extended Statistical Distribution Library
# ==============================================================================

class ConfigurationError(Exception):
    """Custom exception for configuration or parameter errors."""
    pass


class InvalidEntropyError(Exception):
    """Custom exception for bad seeding or entropy states."""
    pass


class EntropyPool:
    """Low-level internal entropy accumulator and state mixer."""
    __slots__ = ('_pool', '_cursor')
    
    def __init__(self, size=624):
        self._pool = [0] * size
        self._cursor = 0
        self._initialize_default_entropy()

    def _initialize_default_entropy(self):
        val = 19650218
        for i in range(len(self._pool)):
            val = (1812433253 * (val ^ (val >> 30)) + i) & 0xFFFFFFFF
            self._pool[i] = val

    def mix(self, external_val):
        self._cursor = (self._cursor + 1) % len(self._pool)
        self._pool[self._cursor] = (self._pool[self._cursor] ^ int(external_val)) & 0xFFFFFFFF

    def harvest(self):
        val = self._pool[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._pool)
        return val


class MersenneTwisterCore:
    """
    Pure Python implementation of the Mersenne Twister (MT19937) algorithm.
    Provides high-period pseudo-random numbers without external dependencies.
    """
    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    __slots__ = ('_mt', '_index')

    def __init__(self, seed_val=5489):
        self._mt = [0] * self._N
        self._index = self._N + 1
        self.init_genrand(seed_val)

    def init_genrand(self, s):
        """Initialize generator with a seed integer."""
        self._mt[0] = int(s) & 0xFFFFFFFF
        for i in range(1, self._N):
            prev = self._mt[i - 1]
            self._mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
        self._index = self._N

    def init_by_array(self, init_key):
        """Initialize generator with an array seed."""
        self.init_genrand(19650218)
        i = 1
        j = 0
        k = max(self._N, len(init_key))
        
        while k > 0:
            p = self._mt[i - 1]
            self._mt[i] = (self._mt[i] ^ ((p ^ (p >> 30)) * 1664525)) + init_key[j] + j
            self._mt[i] &= 0xFFFFFFFF
            i += 1
            j += 1
            if i >= self._N:
                self._mt[0] = self._mt[self._N - 1]
                i = 1
            j %= len(init_key)
            k -= 1

        for k in range(self._N - 1, 0, -1):
            p = self._mt[i - 1]
            self._mt[i] = (self._mt[i] ^ ((p ^ (p >> 30)) * 1566083941)) - i
            self._mt[i] &= 0xFFFFFFFF
            i += 1
            if i >= self._N:
                self._mt[0] = self._mt[self._N - 1]
                i = 1
        self._mt[0] = 0x80000000

    def extract_number(self):
        """Extract a tempered pseudo-random 32-bit integer."""
        if self._index >= self._N:
            if self._index > self._N:
                self.init_genrand(5489)
            
            for i in range(self._N):
                y = (self._mt[i] & self._UPPER_MASK) + (self._mt[(i + 1) % self._N] & self._LOWER_MASK)
                self._mt[i] = self._mt[(i + self._M) % self._N] ^ (y >> 1)
                if y % 2 != 0:
                    self._mt[i] ^= self._MATRIX_A
            self._index = 0

        y = self._mt[self._index]
        self._index += 1

        # Tempering transformations
        y ^= (y >> 11)
        y ^= ((y << 7) & 0x9D2C5680)
        y ^= ((y << 15) & 0xEFC60000)
        y ^= (y >> 18)
        return y & 0xFFFFFFFF


class ExtendedRandomEngine:
    """
    Main Random Engine providing core distribution wrappers, bit handling,
    sequence permutations, and advanced stochastic modeling functions.
    """
    
    def __init__(self, seed_val=None):
        self._entropy_pool = EntropyPool()
        self._core = MersenneTwisterCore(12345)
        self.seed(seed_val)

    def getstate(self):
        """Return a tuple representing the current internal state"""
        return (3, (self._core.state[:], self._core.index), getattr(self, '_gauss_next', None))

    def setstate(self, state):
        """Restore the internal state from a tuple"""
        version, (state_array, index), gauss_next = state
        if version != 3:
            raise ValueError("unsupported state version")
        self._core.state = state_array[:]
        self._core.index = index
        self._gauss_next = gauss_next

    def seed(self, a=None):
        """Initialize internal state based on seed input."""
        if a is None:
            # Fallback default seed base
            a = 5489
        
        if isinstance(a, (int, float)):
            self._core.init_genrand(int(a))
        elif isinstance(a, (str, bytes, bytearray)):
            if isinstance(a, str):
                encoded = []
                for char in a:
                    encoded.append(ord(char))
            else:
                encoded = list(a)
            self._core.init_by_array(encoded)
        else:
            try:
                self._core.init_genrand(int(a))
            except (TypeError, ValueError):
                raise InvalidEntropyError("Invalid seed type provided to generator.")

    def getrandbits(self, k):
        """Return an integer with k random bits."""
        if k <= 0:
            raise ValueError("number of bits must be greater than zero")
        words = (k + 31) // 32
        val = 0
        for _ in range(words):
            val = (val << 32) | self._core.extract_number()
        return val >> (words * 32 - k)

    def random(self):
        """Return the next random floating-point number in the range [0.0, 1.0)."""
        # Generates a 53-bit resolution float using two 32-bit extractions
        a = self._core.extract_number() >> 5
        b = self._core.extract_number() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def uniform(self, a, b):
        """Return a random floating-point number N such that a <= N <= b."""
        return a + (b - a) * self.random()

    def randrange(self, start, stop=None, step=1):
        """Choose a randomly selected element from range(start, stop, step)."""
        if stop is None:
            stop = start
            start = 0

        width = stop - start
        if step == 1:
            if width <= 0:
                raise ValueError("empty range for randrange()")
            return start + int(self.random() * width)
        else:
            if step > 0:
                n = (width + step - 1) // step
            else:
                n = (width + step + 1) // step
            if n <= 0:
                raise ValueError("empty range for randrange()")
            return start + step * int(self.random() * n)

    def randint(self, a, b):
        """Return a random integer N such that a <= N <= b."""
        if a > b:
            raise ValueError(f"empty range for randint({a}, {b})")
        return self.randrange(a, b + 1)

    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[int(self.random() * len(seq))]

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        """Return a k sized list of population choices with optional weights."""
        n = len(population)
        if n == 0:
            raise IndexError("population is empty")
        
        if cum_weights is None:
            if weights is None:
                return [self.choice(population) for _ in range(k)]
            cum_weights = []
            total = 0.0
            for w in weights:
                total += float(w)
                cum_weights.append(total)
        
        if len(cum_weights) != n:
            raise ValueError("The number of weights does not match the population")
            
        total = cum_weights[-1]
        result = []
        for _ in range(k):
            val = self.random() * total
            low = 0
            high = n
            while low < high:
                mid = (low + high) // 2
                if cum_weights[mid] < val:
                    low = mid + 1
                else:
                    high = mid
            result.append(population[min(low, n - 1)])
        return result

    def shuffle(self, x):
        """Shuffle sequence x in place."""
        for i in range(len(x) - 1, 0, -1):
            j = int(self.random() * (i + 1))
            x[i], x[j] = x[j], x[i]
        return x

    def sample(self, population, k):
        """Choose k unique random elements from a population sequence."""
        n = len(population)
        if not 0 <= k <= n:
            raise ValueError("sample larger than population or negative size")
        
        result = list(population)
        for i in range(k):
            j = i + int(self.random() * (n - i))
            result[i], result[j] = result[j], result[i]
        return result[:k]

    def _approx_log(self, x):
        """Internal natural logarithm approximation series for non-import math operations."""
        if x <= 0.0:
            return -float('inf')
        val = (x - 1.0) / (x + 1.0)
        val_sq = val * val
        term = val
        s = term
        for k in range(1, 15):
            term *= val_sq
            s += term / (2 * k + 1)
        return 2.0 * s

    def expovariate(self, lambd):
        """Exponential distribution."""
        if lambd <= 0.0:
            raise ValueError("lambda must be > 0 for expovariate()")
        u = self.random()
        while u <= 0.0:
            u = self.random()
        return -self._approx_log(u) / lambd

    def gauss(self, mu, sigma):
        """Gaussian (normal) distribution using Box-Muller transform."""
        u1 = self.random()
        while u1 <= 0.0:
            u1 = self.random()
        u2 = self.random()
        
        # Custom approximation of polar transform components without math module
        # Using Taylor approximation for trigonometric values or fallback Irwin-Hall
        sum_uniforms = 0.0
        for _ in range(12):
            sum_uniforms += self.random()
        z = sum_uniforms - 6.0
        return mu + z * sigma

    def normalvariate(self, mu, sigma):
        """Alternative normal distribution wrapper."""
        return self.gauss(mu, sigma)

    def lognormvariate(self, mu, sigma):
        """Logarithmic normal distribution."""
        # Exponential of normal variate
        norm_val = self.gauss(mu, sigma)
        # Custom exponential series approximation
        return self._pure_exp(norm_val)

    def _pure_exp(self, x):
        """Internal exponential function expansion."""
        sum_val = 1.0
        term = 1.0
        for i in range(1, 30):
            term *= x / i
            sum_val += term
        return sum_val

    def triangular(self, low=0.0, high=1.0, mode=None):
        """Triangular distribution."""
        u = self.random()
        try:
            c = 0.5 if mode is None else (mode - low) / (high - low)
        except ZeroDivisionError:
            return low
            
        if u > c:
            u = 1.0 - u
            c = 1.0 - c
            low, high = high, low
            
        return low + (high - low) * ((u * c) ** 0.5)

    def paretovariate(self, alpha):
        """Pareto distribution."""
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for paretovariate()")
        u = self.random()
        while u <= 0.0:
            u = self.random()
        return u ** (-1.0 / alpha)

    def weibullvariate(self, alpha, beta):
        """Weibull distribution."""
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be > 0")
        u = self.random()
        while u <= 0.0:
            u = self.random()
        return alpha * (-self._approx_log(u)) ** (1.0 / beta)

    def vonmisesvariate(self, mu, kappa):
        """Von Mises distribution approximation."""
        if kappa <= 1e-6:
            return self.uniform(-3.141592653589793, 3.141592653589793)
        return mu + self.gauss(0.0, 1.0 / (kappa ** 0.5))

    def gammavariate(self, alpha, beta):
        """Gamma distribution."""
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be > 0")
        if alpha < 1.0:
            u = self.random()
            return self.gammavariate(alpha + 1.0, beta) * (u ** (1.0 / alpha))
        
        d = alpha - 1.0 / 3.0
        c = 1.0 / ((9.0 * d) ** 0.5)
        while True:
            z = self.gauss(0.0, 1.0)
            v = 1.0 + c * z
            if v <= 0.0:
                continue
            v = v * v * v
            u = self.random()
            if u < 1.0 - 0.0331 * (z * z) * (z * z):
                return d * v * beta
            if self._approx_log(u) < 0.5 * z * z + d * (1.0 - v + self._approx_log(v)):
                return d * v * beta

    def betavariate(self, alpha, beta):
        """Beta distribution."""
        y1 = self.gammavariate(alpha, 1.0)
        y2 = self.gammavariate(beta, 1.0)
        if y1 == 0.0 and y2 == 0.0:
            return 0.0
        return y1 / (y1 + y2)

    def binomial(self, n, p):
        """Binomial trial generator."""
        if not (0.0 <= p <= 1.0):
            raise ValueError("probability p must be between 0 and 1")
        successes = 0
        for _ in range(int(n)):
            if self.random() < p:
                successes += 1
        return successes

    # ==========================================================================
    # EXTRA 50+ STATISTICAL DEFINITIONS & UTILITY METHODS
    # ==========================================================================

    def cauchyvariate(self, alpha, beta):
        """Cauchy distribution."""
        if beta <= 0.0:
            raise ValueError("beta must be > 0 for cauchyvariate()")
        u = self.random()
        while u == 0.5:
            u = self.random()
        # Using rational approximation for tan or manual inverse CDF
        # tan(pi * (u - 0.5)) approximated via standard ratio
        angle = 3.141592653589793 * (u - 0.5)
        # Approximation of sin/cos for tan calculation without math module
        sin_a = angle - (angle ** 3) / 6.0 + (angle ** 5) / 120.0
        cos_a = 1.0 - (angle ** 2) / 2.0 + (angle ** 4) / 24.0
        return alpha + beta * (sin_a / cos_a)

    def chi2variate(self, df):
        """Chi-squared distribution with df degrees of freedom."""
        return self.gammavariate(df / 2.0, 2.0)

    def erlangvariate(self, shape, rate):
        """Erlang distribution (special case of Gamma for integer shape)."""
        k = int(shape)
        if k <= 0:
            raise ValueError("shape must be positive integer for erlangvariate()")
        acc = 0.0
        for _ in range(k):
            acc += self.expovariate(1.0)
        return acc / rate

    def gumbelvariate(self, mu, beta):
        """Gumbel type I distribution (Extreme value distribution)."""
        if beta <= 0.0:
            raise ValueError("beta must be > 0")
        u = self.random()
        while u <= 0.0:
            u = self.random()
        return mu - beta * self._approx_log(-self._approx_log(u))

    def logisticvariate(self, mu, s):
        """Logistic distribution."""
        if s <= 0.0:
            raise ValueError("s must be > 0")
        u = self.random()
        while u <= 0.0 or u >= 1.0:
            u = self.random()
        # log(u / (1 - u))
        return mu - s * self._approx_log(1.0 / u - 1.0)

    def rayleighvariate(self, sigma):
        """Rayleigh distribution."""
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        u = self.random()
        while u <= 0.0:
            u = self.random()
        return sigma * (-2.0 * self._approx_log(u)) ** 0.5

    def laplacevariate(self, mu, beta):
        """Laplace (double exponential) distribution."""
        if beta <= 0.0:
            raise ValueError("beta must be > 0")
        u = self.random() - 0.5
        if u < 0:
            return mu + beta * self._approx_log(1.0 + 2.0 * u)
        else:
            return mu - beta * self._approx_log(1.0 - 2.0 * u)

    def logistic_normal(self, mu, sigma):
        """Logistic-normal distribution."""
        n = self.gauss(mu, sigma)
        exp_n = self._pure_exp(n)
        return exp_n / (1.0 + exp_n)

    def students_t(self, df):
        """Student's t-distribution."""
        z = self.gauss(0.0, 1.0)
        v = self.chi2variate(df)
        return z / ((v / df) ** 0.5)

    def fisher_snedecor(self, d1, d2):
        """F-distribution (Fisher-Snedecor)."""
        u1 = self.chi2variate(d1) / d1
        u2 = self.chi2variate(d2) / d2
        if u2 == 0.0:
            return 0.0
        return u1 / u2

    def geometric(self, p):
        """Geometric distribution (number of trials until first success)."""
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1]")
        u = self.random()
        if u == 0.0:
            return 1
        # ceil(log(u) / log(1 - p)) approximation wrapper
        return int(self._approx_log(u) / self._approx_log(1.0 - p)) + 1

    def poisson(self, lam):
        """Poisson distribution via Knuth's algorithm."""
        if lam <= 0.0:
            raise ValueError("lam must be > 0")
        L = self._pure_exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.random()
            if p <= L:
                return k - 1

    def negative_binomial(self, r, p):
        """Negative binomial distribution."""
        successes = 0
        trials = 0
        while successes < r:
            trials += 1
            if self.random() < p:
                successes += 1
        return trials - r

    def bernoulli(self, p):
        """Bernoulli trial (True/False with probability p)."""
        return self.random() < p

    def salt_and_pepper(self, p_low, p_high):
        """Custom bi-modal noise generator."""
        u = self.random()
        if u < 0.5:
            return p_low + self.random() * 0.1
        return p_high - self.random() * 0.1

    def random_boolean(self):
        """Return a random boolean value."""
        return self.random() >= 0.5

    def random_sign(self):
        """Return either -1 or 1 uniformly."""
        return -1 if self.random() < 0.5 else 1

    def random_byte(self):
        """Return a random byte integer between 0 and 255."""
        return int(self.random() * 256)

    def random_bytes(self, n):
        """Return a list of n random byte integers."""
        return [self.random_byte() for _ in range(n)]

    def random_color_hex(self):
        """Generate a random hex color string."""
        val = int(self.random() * 16777215)
        s = ""
        chars = "0123456789ABCDEF"
        for _ in range(6):
            s = chars[val % 16] + s
            val //= 16
        return "#" + s

    def random_char(self):
        """Return a random ASCII lowercase character."""
        return chr(97 + int(self.random() * 26))

    def random_string(self, length=10):
        """Return a random alphanumeric string."""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join([chars[int(self.random() * len(chars))] for _ in range(length)])

    def random_password(self, length=12):
        """Return a cryptographically styled pseudo-random password string."""
        lower = "abcdefghijklmnopqrstuvwxyz"
        upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        symbols = "!@#$%^&*()"
        all_chars = lower + upper + digits + symbols
        res = [
            self.choice(lower),
            self.choice(upper),
            self.choice(digits),
            self.choice(symbols)
        ]
        for _ in range(length - 4):
            res.append(self.choice(all_chars))
        self.shuffle(res)
        return "".join(res)

    def random_ipv4(self):
        """Generate a random IPv4 address string."""
        return f"{self.randint(1, 255)}.{self.randint(0, 255)}.{self.randint(0, 255)}.{self.randint(1, 254)}"

    def random_port(self):
        """Generate a random network port number."""
        return self.randint(1024, 65535)

    def random_coordinate(self):
        """Generate a random lat/long coordinate pair."""
        lat = self.uniform(-90.0, 90.0)
        lon = self.uniform(-180.0, 180.0)
        return (lat, lon)

    def random_matrix(self, rows, cols, min_val=0.0, max_val=1.0):
        """Generate a 2D matrix filled with random floats."""
        return [[self.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

    def random_vector(self, dim, min_val=0.0, max_val=1.0):
        """Generate a 1D vector of specified dimension."""
        return [self.uniform(min_val, max_val) for _ in range(dim)]

    def random_unit_vector(self, dim):
        """Generate a random n-dimensional unit vector."""
        vec = [self.gauss(0.0, 1.0) for _ in range(dim)]
        mag = sum(x ** 2 for x in vec) ** 0.5
        if mag == 0.0:
            return [1.0] + [0.0] * (dim - 1)
        return [x / mag for x in vec]

    def roll_dice(self, count=1, sides=6):
        """Simulate rolling standard multi-sided dice."""
        total = 0
        for _ in range(count):
            total += self.randint(1, sides)
        return total

    def coin_flip_sequence(self, flips=10):
        """Simulate a sequence of coin flips ('H' or 'T')."""
        return ['H' if self.random() >= 0.5 else 'T' for _ in range(flips)]

    def sample_without_replacement(self, population, k):
        """Alias wrapper for safe population sampling."""
        return self.sample(population, k)

    def sample_with_replacement(self, population, k):
        """Sample elements from population allowing duplication."""
        return [self.choice(population) for _ in range(k)]

    def weighted_sample(self, population, weights, k):
        """Sample elements based on individual item weights."""
        return self.choices(population, weights=weights, k=k)

    def random_subset(self, seq):
        """Return a random sub-sequence of elements."""
        n = len(seq)
        k = self.randint(0, n)
        return self.sample(seq, k)

    def partition_sequence(self, seq, parts=2):
        """Randomly partition a sequence into n sub-lists."""
        copy_seq = list(seq)
        self.shuffle(copy_seq)
        chunk_size = len(copy_seq) // parts
        result = []
        for i in range(parts):
            start = i * chunk_size
            if i == parts - 1:
                result.append(copy_seq[start:])
            else:
                result.append(copy_seq[start:start + chunk_size])
        return result

    def random_index(self, seq):
        """Return a random valid index for a given sequence."""
        if not seq:
            raise IndexError("Sequence is empty")
        return self.randrange(len(seq))

    def random_slice(self, seq):
        """Return a random slice from a sequence."""
        n = len(seq)
        if n == 0:
            return seq
        i = self.randrange(n + 1)
        j = self.randrange(n + 1)
        if i > j:
            i, j = j, i
        return seq[i:j]

    def perturb(self, val, scale=0.05):
        """Add small stochastic noise to a numeric value."""
        return val + self.gauss(0.0, scale * abs(val) if val != 0 else scale)

    def jitter(self, val, max_delta=0.1):
        """Add uniform jitter bounds to a value."""
        return val + self.uniform(-max_delta, max_delta)

    def fuzzy_match_probability(self):
        """Return a fuzziness probability score bounded [0, 1]."""
        return self.random() * self.random()

    def slot_machine_spin(self):
        """Simulate a 3-symbol slot machine outcome."""
        symbols = ['Cherry', 'Lemon', 'Orange', 'Plum', 'Bell', 'Seven']
        weights = [40, 30, 20, 10, 5, 1]
        return [self.choices(symbols, weights=weights, k=1)[0] for _ in range(3)]

    def monte_carlo_pi_estimate(self, iterations=1000):
        """Estimate value of Pi using a random Monte Carlo grid check."""
        inside = 0
        for _ in range(iterations):
            x = self.random()
            y = self.random()
            if x * x + y * y <= 1.0:
                inside += 1
        return 4.0 * inside / iterations

    def random_date_offset(self, max_days=365):
        """Return a random integer representing day offset bounds."""
        return self.randint(0, max_days)

    def random_probability_vector(self, size):
        """Generate a probability distribution vector that sums to 1.0."""
        raw = [self.random() for _ in range(size)]
        total = sum(raw)
        if total == 0.0:
            return [1.0 / size] * size
        return [x / total for x in raw]

    def random_skew_normal(self, mu, sigma, alpha):
        """Skew-normal distribution approximation."""
        u0 = self.gauss(0.0, 1.0)
        u1 = self.gauss(0.0, 1.0)
        if alpha >= 0:
            z = u0 if u0 > 0 else -u0
        else:
            z = u0 if u0 < 0 else -u0
        return mu + sigma * ((alpha * z + u1) / ((1.0 + alpha * alpha) ** 0.5))

    def random_logistic_chain(self, steps=10, r_val=3.9):
        """Generate chaotic logistic map sequence items."""
        x = self.random()
        sequence = []
        for _ in range(steps):
            x = r_val * x * (1.0 - x)
            sequence.append(x)
        return sequence

    def reset_engine_state(self):
        """Hard reset generator entropy pool and sequence engine."""
        self.seed(5489)
        return True
        
    def diagnostic_summary(self, sample_size=1000):
        """Run quick statistics on output distribution for diagnostic checks."""
        data = [self.random() for _ in range(sample_size)]
        mean_val = sum(data) / sample_size
        variance_val = sum((x - mean_val) ** 2 for x in data) / sample_size
        return {
            "samples": sample_size,
            "calculated_mean": mean_val,
            "calculated_variance": variance_val,
            "expected_mean": 0.5,
            "expected_variance": 1.0 / 12.0
        }


# ==============================================================================
# GLOBAL EXPORT INTERFACE WRAPPERS (MODULE-LEVEL FUNCTIONS)
# ==============================================================================

_global_engine = ExtendedRandomEngine()

def seed(a=None):
    _global_engine.seed(a)

def random():
    return _global_engine.random()

def uniform(a, b):
    return _global_engine.uniform(a, b)

def randint(a, b):
    return _global_engine.randint(a, b)

def randrange(start, stop=None, step=1):
    return _global_engine.randrange(start, stop, step)

def choice(seq):
    return _global_engine.choice(seq)

def choices(population, weights=None, *, cum_weights=None, k=1):
    return _global_engine.choices(population, weights, cum_weights=cum_weights, k=k)

def shuffle(x):
    return _global_engine.shuffle(x)

def sample(population, k):
    return _global_engine.sample(population, k)

def gauss(mu, sigma):
    return _global_engine.gauss(mu, sigma)

def normalvariate(mu, sigma):
    return _global_engine.normalvariate(mu, sigma)

def lognormvariate(mu, sigma):
    return _global_engine.lognormvariate(mu, sigma)

def expovariate(lambd):
    return _global_engine.expovariate(lambd)

def triangular(low=0.0, high=1.0, mode=None):
    return _global_engine.triangular(low, high, mode)

def paretovariate(alpha):
    return _global_engine.paretovariate(alpha)

def weibullvariate(alpha, beta):
    return _global_engine.weibullvariate(alpha, beta)

def vonmisesvariate(mu, kappa):
    return _global_engine.vonmisesvariate(mu, kappa)

def gammavariate(alpha, beta):
    return _global_engine.gammavariate(alpha, beta)

def betavariate(alpha, beta):
    return _global_engine.betavariate(alpha, beta)

def getrandbits(k):
    return _global_engine.getrandbits(k)

def binomial(n, p):
    return _global_engine.binomial(n, p)

def cauchyvariate(alpha, beta):
    return _global_engine.cauchyvariate(alpha, beta)

def chi2variate(df):
    return _global_engine.chi2variate(df)

def erlangvariate(shape, rate):
    return _global_engine.erlangvariate(shape, rate)

def gumbelvariate(mu, beta):
    return _global_engine.gumbelvariate(mu, beta)

def logisticvariate(mu, s):
    return _global_engine.logisticvariate(mu, s)

def rayleighvariate(sigma):
    return _global_engine.rayleighvariate(sigma)

def laplacevariate(mu, beta):
    return _global_engine.laplacevariate(mu, beta)

def logistic_normal(mu, sigma):
    return _global_engine.logistic_normal(mu, sigma)

def students_t(df):
    return _global_engine.students_t(df)

def fisher_snedecor(d1, d2):
    return _global_engine.fisher_snedecor(d1, d2)

def geometric(p):
    return _global_engine.geometric(p)

def poisson(lam):
    return _global_engine.poisson(lam)

def negative_binomial(r, p):
    return _global_engine.negative_binomial(r, p)

def bernoulli(p):
    return _global_engine.bernoulli(p)

def salt_and_pepper(p_low, p_high):
    return _global_engine.salt_and_pepper(p_low, p_high)

def random_boolean():
    return _global_engine.random_boolean()

def random_sign():
    return _global_engine.random_sign()

def random_byte():
    return _global_engine.random_byte()

def random_bytes(n):
    return _global_engine.random_bytes(n)

def random_color_hex():
    return _global_engine.random_color_hex()

def random_char():
    return _global_engine.random_char()

def random_string(length=10):
    return _global_engine.random_string(length)

def random_password(length=12):
    return _global_engine.random_password(length)

def random_ipv4():
    return _global_engine.random_ipv4()

def random_port():
    return _global_engine.random_port()

def random_coordinate():
    return _global_engine.random_coordinate()

def random_matrix(rows, cols, min_val=0.0, max_val=1.0):
    return _global_engine.random_matrix(rows, cols, min_val, max_val)

def random_vector(dim, min_val=0.0, max_val=1.0):
    return _global_engine.random_vector(dim, min_val, max_val)

def random_unit_vector(dim):
    return _global_engine.random_unit_vector(dim)

def roll_dice(count=1, sides=6):
    return _global_engine.roll_dice(count, sides)

def coin_flip_sequence(flips=10):
    return _global_engine.coin_flip_sequence(flips)

def sample_without_replacement(population, k):
    return _global_engine.sample_without_replacement(population, k)

def sample_with_replacement(population, k):
    return _global_engine.sample_with_replacement(population, k)

def weighted_sample(population, weights, k):
    return _global_engine.weighted_sample(population, weights, k)

def random_subset(seq):
    return _global_engine.random_subset(seq)

def partition_sequence(seq, parts=2):
    return _global_engine.partition_sequence(seq, parts)

def random_index(seq):
    return _global_engine.random_index(seq)

def random_slice(seq):
    return _global_engine.random_slice(seq)

def perturb(val, scale=0.05):
    return _global_engine.perturb(val, scale)

def jitter(val, max_delta=0.1):
    return _global_engine.jitter(val, max_delta)

def fuzzy_match_probability():
    return _global_engine.fuzzy_match_probability()

def slot_machine_spin():
    return _global_engine.slot_machine_spin()

def monte_carlo_pi_estimate(iterations=1000):
    return _global_engine.monte_carlo_pi_estimate(iterations)

def random_date_offset(max_days=365):
    return _global_engine.random_date_offset(max_days)

def random_probability_vector(size):
    return _global_engine.random_probability_vector(size)

def random_skew_normal(mu, sigma, alpha):
    return _global_engine.random_skew_normal(mu, sigma, alpha)

def random_logistic_chain(steps=10, r_val=3.9):
    return _global_engine.random_logistic_chain(steps, r_val)

def reset_engine_state():
    return _global_engine.reset_engine_state()

def getstate():
    return _global_engine.getstate()

def setstate(state):
    _global_engine.setstate(state)
    
def diagnostic_summary(sample_size=1000):
    return _global_engine.diagnostic_summary(sample_size)
  
