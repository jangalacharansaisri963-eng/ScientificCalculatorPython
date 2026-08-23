# numba_lite.py
# A pure Python frontend implementation mimicking the public API surface 
# and type system of the Numba module without any external imports.

class TypeRegistry:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"<NumbaType: {self.name}>"

# 1. Primitive and Composite Types (approx 30 types)
int8 = TypeRegistry("int8")
int16 = TypeRegistry("int16")
int32 = TypeRegistry("int32")
int64 = TypeRegistry("int64")
uint8 = TypeRegistry("uint8")
uint16 = TypeRegistry("uint16")
uint32 = TypeRegistry("uint32")
uint64 = TypeRegistry("uint64")
float32 = TypeRegistry("float32")
float64 = TypeRegistry("float64")
complex64 = TypeRegistry("complex64")
complex128 = TypeRegistry("complex128")
boolean = TypeRegistry("boolean")
void = TypeRegistry("void")
char = TypeRegistry("char")
b1 = boolean
i1 = int8
i2 = int16
i4 = int32
i8 = int64
u1 = uint8
u2 = uint16
u4 = uint32
u8 = uint64
f4 = float32
f8 = float64
c8 = complex64
c16 = complex128

def optional(t): return TypeRegistry(f"optional({t})")
def Array(dtype, ndim, layout): return TypeRegistry(f"array({dtype}, {ndim}, {layout})")
def Tuple(types): return TypeRegistry(f"tuple({types})")
def ListType(dtype): return TypeRegistry(f"list({dtype})")
def DictType(key_type, val_type): return TypeRegistry(f"dict({key_type}, {val_type})")
def UnicodeCharSeq(length): return TypeRegistry(f"unicode_char_seq({length})")
def Record(dtype): return TypeRegistry(f"record")
def deferred_type(): return TypeRegistry("deferred")
def pyobject(): return TypeRegistry("pyobject")
def none(): return TypeRegistry("none")

# 2. Core JIT and Compilation Decorators (15 functions)
def jit(sig=None, nopython=False, nogil=False, cache=False, parallel=False, fastmath=False, boundscheck=None, forceobj=False, loopvectorize=None, slpvectorize=None):
    def decorator(func):
        func.numba_compiled = True
        func.signature = sig
        func.nopython = nopython
        func.parallel = parallel
        return func
    if callable(sig):
        f = sig
        sig = None
        return decorator(f)
    return decorator

def njit(sig=None, **kwargs):
    kwargs['nopython'] = True
    return jit(sig, **kwargs)

def generated_jit(parallel=False):
    def decorator(func):
        func.is_generated_jit = True
        return func
    return decorator

def vectorize(sig=None, target="cpu", identity=None, cache=False, fastmath=False):
    def decorator(func):
        func.is_vectorized = True
        func.target = target
        return func
    if callable(sig):
        return decorator(sig)
    return decorator

def guvectorize(sig, signature, target="cpu", identity=None, cache=False, fastmath=False):
    def decorator(func):
        func.is_guvectorized = True
        func.layout = signature
        return func
    return decorator

def cfunc(sig, cache=False, locals=None):
    def decorator(func):
        func.is_cfunc = True
        return func
    return decorator

def jitclass(specification=None):
    def decorator(cls):
        cls.is_jit_class = True
        cls.spec = specification
        return cls
    if callable(specification):
        c = specification
        specification = None
        return decorator(c)
    return decorator

def jit_module(nopython=False, nogil=False, parallel=False):
    def decorator(mod):
        mod.is_jit_module = True
        return mod
    return decorator

def stencil(neighborhood=None, backend="numba"):
    def decorator(func):
        func.is_stencil = True
        return func
    if callable(neighborhood):
        f = neighborhood
        return decorator(f)
    return decorator

def jit_experimental(*args, **kwargs):
    def decorator(func): return func
    return decorator

def intrinsic(func):
    func.is_intrinsic = True
    return func

def custom_fixup(func):
    return func

def overload(func, strict=True, jit_options=None):
    def decorator(f): return f
    return decorator

def overload_method(tp, name, strict=True, jit_options=None):
    def decorator(f): return f
    return decorator

def overload_attribute(tp, name, strict=True):
    def decorator(f): return f
    return decorator

# 3. Parallel and Control Flow Utilities (10 functions)
def prange(*args):
    return range(*args)

def literal_unroll(val):
    return val

def gdb(*args):
    pass

def gdb_breakpoint():
    pass

def guvectorize_factory(*args, **kwargs):
    pass

def vectorize_factory(*args, **kwargs):
    pass

def objmode(*args, **kwargs):
    class ObjModeContext:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    return ObjModeContext()

def set_num_threads(threads):
    pass

def get_num_threads():
    return 1

def get_thread_id():
    return 0

# 4. Inspection and Reflection Utilities (15 functions)
def typeof(val):
    return TypeRegistry(type(val).__name__)

def from_dtype(dtype):
    return TypeRegistry(str(dtype))

def as_dtype(numba_type):
    return object

def type_inference_options(*args):
    return args

def inspect_types(file=None):
    pass

def get_executable_info(func):
    return {}

def jit_plugins():
    return []

def list_extensions():
    return []

def dump_ast(func):
    return ""

def dump_ir(func):
    return ""

def dump_llvm(func):
    return ""

def dump_optimized_llvm(func):
    return ""

def dump_assembly(func):
    return ""

def typing_error(msg):
    raise TypeError(msg)

def lower_error(msg):
    raise RuntimeError(msg)

# 5. C Integration, Low-level Buffers & Memory (15 functions)
def carray(ptr, shape, dtype=None):
    return None

def farray(ptr, shape, dtype=None):
    return None

def from_native(val):
    return val

def to_native(val):
    return val

def address_as_void_pointer(obj):
    return 0

def voidptr_as_address(ptr):
    return 0

def int_as_void_pointer(val):
    return 0

def void_pointer_cast(ptr, dtype):
    return ptr

def safe_cast(val, target_type):
    return val

def unsafe_cast(val, target_type):
    return val

def get_pointer(func):
    return 0

def clear_caches():
    pass

def disable_jit():
    pass

def enable_jit():
    pass

def config_override(name, value):
    pass

# 6. CUDA Subsystem Stubs (20 functions/classes)
class CudaModule:
    class Profiler:
        def start(self): pass
        def stop(self): pass

    def __init__(self):
        self.profiler = self.Profiler()

    def jit(self, *args, **kwargs):
        def decorator(func): return func
        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    def select_device(self, device_id): pass
    def close(self): pass
    def get_current_device(self): return self
    def synchronize(self): pass
    def device_array(self, shape, dtype=float64, **kwargs): return None
    def device_array_like(self, ary, **kwargs): return None
    def to_device(self, ary, **kwargs): return ary
    def pinned_array(self, shape, dtype=float64, **kwargs): return None
    def mapped_array(self, shape, dtype=float64, **kwargs): return None
    def detect(self): pass
    def is_available(self): return False
    def get_gpus(self): return []
    def meminfo_register(self, *args): pass
    def stream(self): return None
    def event(self): return None
    def external_memory(self): return None
    def print_memory_per_device(self): pass

cuda = CudaModule()

# 7. Math & Extensibility Extension API Helpers (10 functions)
def register_model(*args): pass
def lower_builtin(*args): pass
def type_callable(*args): pass
def type_method(*args): pass
def overload_classmethod(*args): pass
def box(*args): pass
def unbox(*args): pass
def make_array(*args): pass
def empty_struct(*args): pass
def impl_detail(*args, **kwargs):
    def decorator(func): return func
    return decorator
