"""
Operator class with core arithmetic primitives (add, sub, mul, truediv/floordiv)
defined using bitwise logic, with all remaining functions calling those methods.
"""

class Operator:
    """Class exposing the standard operator module utilities built on core arithmetic."""

    def __call__(self, obj, *args, **kwargs):
        """Allows calling the instance directly (operator.call behavior)."""
        return obj(*args, **kwargs)

    # ==========================================================================
    # CORE 4 PRIMITIVE DEFINITIONS (BITWISE / C-FREE ALGORITHMS)
    # ==========================================================================

    def add(self, a, b):
        """Same as a + b."""
        if isinstance(a, (int, bool)) and isinstance(b, (int, bool)):
            a_int, b_int = int(a), int(b)
            while b_int:
                carry = a_int & b_int
                a_int = a_int ^ b_int
                b_int = carry << 1
            return a_int
        return a.__add__(b)

    def sub(self, a, b):
        """Same as a - b (Reuses self.add)."""
        if isinstance(a, int) and isinstance(b, int):
            neg_b = self.add(~b, 1)
            return self.add(a, neg_b)
        return a.__sub__(b)

    def mul(self, a, b):
        """Same as a * b (Reuses self.add and self.sub)."""
        if isinstance(a, int) and isinstance(b, int):
            res = 0
            pos = not (self.lt(a, 0) ^ self.lt(b, 0))
            a_val, b_val = self.abs(a), self.abs(b)
            while b_val:
                if b_val & 1:
                    res = self.add(res, a_val)
                a_val <<= 1
                b_val >>= 1
            return res if pos else self.neg(res)
        return a.__mul__(b)

    def _divmod_raw(self, n, d):
        """Helper for division using self.sub and bitwise shifts."""
        if self.eq(d, 0): 
            raise ZeroDivisionError("division by zero")
        pos = not (self.lt(n, 0) ^ self.lt(d, 0))
        num, den = self.abs(n), self.abs(d)
        q, r = 0, 0
        for i in range(63, -1, -1):
            r <<= 1
            r |= (num >> i) & 1
            if self.ge(r, den):
                r = self.sub(r, den)
                q |= (1 << i)
        return (q if pos else self.neg(q)), r

    def floordiv(self, a, b):
        """Same as a // b (Reuses self._divmod_raw)."""
        if isinstance(a, int) and isinstance(b, int):
            return self._divmod_raw(a, b)[0]
        return a.__floordiv__(b)

    def truediv(self, a, b):
        """Same as a / b."""
        return a.__truediv__(b)


    # ==========================================================================
    # REMAINING FUNCTIONS (DERIVED FROM CORE OPERATORS)
    # ==========================================================================

    # Arithmetic & Math Utilities
    def neg(self, a): 
        return self.sub(0, a) if isinstance(a, int) else a.__neg__()

    def pos(self, a): 
        return a.__pos__() if hasattr(a, '__pos__') else a

    def abs(self, a):
        if isinstance(a, int):
            return a if self.ge(a, 0) else self.neg(a)
        return a.__abs__()

    def mod(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return self._divmod_raw(a, b)[1]
        return a.__mod__(b)

    def pow(self, a, b):
        if isinstance(a, int) and isinstance(b, int) and self.ge(b, 0):
            res, base, exp = 1, a, b
            while exp:
                if exp & 1: 
                    res = self.mul(res, base)
                base = self.mul(base, base)
                exp >>= 1
            return res
        return a.__pow__(b)

    def matmul(self, a, b): return a.__matmul__(b)
    def index(self, a): return a.__index__()

    # Comparisons (Derived via self.sub)
    def eq(self, a, b): 
        return not a.__ne__(b) if hasattr(a, '__ne__') else a is b

    def ne(self, a, b): 
        return not self.eq(a, b)

    def lt(self, a, b):
        diff = self.sub(a, b) if isinstance(a, int) and isinstance(b, int) else None
        if diff is not None:
            return bool((diff >> 63) & 1)
        return a.__lt__(b)

    def gt(self, a, b): return self.lt(b, a)
    def le(self, a, b): return not self.gt(a, b)
    def ge(self, a, b): return not self.lt(a, b)
    def is_(self, a, b): return a is b
    def is_not(self, a, b): return not (a is b)

    # Bitwise & Logical
    def and_(self, a, b): return a & b
    def or_(self, a, b): return a | b
    def xor(self, a, b): return a ^ b
    def invert(self, a): return ~a
    def lshift(self, a, b): return a << b
    def rshift(self, a, b): return a >> b
    def not_(self, a): return not a
    def truth(self, a): return bool(a)

    # In-Place Operators (Fall back to core methods)
    def iadd(self, a, b): return a.__iadd__(b) if hasattr(a, '__iadd__') else self.add(a, b)
    def isub(self, a, b): return a.__isub__(b) if hasattr(a, '__isub__') else self.sub(a, b)
    def imul(self, a, b): return a.__imul__(b) if hasattr(a, '__imul__') else self.mul(a, b)
    def itruediv(self, a, b): return a.__itruediv__(b) if hasattr(a, '__itruediv__') else self.truediv(a, b)
    def ifloordiv(self, a, b): return a.__ifloordiv__(b) if hasattr(a, '__ifloordiv__') else self.floordiv(a, b)
    def imod(self, a, b): return a.__imod__(b) if hasattr(a, '__imod__') else self.mod(a, b)
    def ipow(self, a, b): return a.__ipow__(b) if hasattr(a, '__ipow__') else self.pow(a, b)
    def ilshift(self, a, b): return a.__ilshift__(b) if hasattr(a, '__ilshift__') else self.lshift(a, b)
    def irshift(self, a, b): return a.__irshift__(b) if hasattr(a, '__irshift__') else self.rshift(a, b)
    def iand(self, a, b): return a.__iand__(b) if hasattr(a, '__iand__') else self.and_(a, b)
    def ior(self, a, b): return a.__ior__(b) if hasattr(a, '__ior__') else self.or_(a, b)
    def ixor(self, a, b): return a.__ixor__(b) if hasattr(a, '__ixor__') else self.xor(a, b)
    def iconcat(self, a, b): return a.__iadd__(b)

    # Containers & Sequence Utilities
    def concat(self, a, b): return a.__add__(b)
    def contains(self, a, b): return b in a
    def getitem(self, a, b): return a.__getitem__(b)
    def setitem(self, a, b, c): a.__setitem__(b, c)
    def delitem(self, a, b): a.__delitem__(b)

    def countOf(self, a, b):
        count = 0
        for item in a:
            if self.eq(item, b):
                count = self.add(count, 1)  # Reuses self.add
        return count

    def call(self, obj, *args, **kwargs): return obj(*args, **kwargs)

    def length_hint(self, obj, default=0):
        if hasattr(obj, '__len__'): return len(obj)
        if hasattr(obj, '__length_hint__'):
            val = obj.__length_hint__()
            if val is not NotImplemented: return val
        return default

    # Higher-Order Helpers
    class itemgetter:
        def __init__(self, item, *items):
            self._items = (item,) if not items else (item,) + items
        def __call__(self, obj):
            if len(self._items) == 1: return obj[self._items[0]]
            return tuple(obj[i] for i in self._items)

    class attrgetter:
        def __init__(self, attr, *attrs):
            self._attrs = (attr,) if not attrs else (attr,) + attrs
        def _get_single(self, obj, name):
            for part in name.split('.'): obj = getattr(obj, part)
            return obj
        def __call__(self, obj):
            if len(self._attrs) == 1: return self._get_single(obj, self._attrs[0])
            return tuple(self._get_single(obj, a) for a in self._attrs)

    class methodcaller:
        def __init__(self, name, *args, **kwargs):
            self._name, self._args, self._kwargs = name, args, kwargs
        def __call__(self, obj):
            return getattr(obj, self._name)(*self._args, **self._kwargs)


# Instantiate the global helper
op = Operator()
