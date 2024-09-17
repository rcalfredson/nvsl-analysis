import pkgutil

has_numba = pkgutil.find_loader("numba")
if has_numba:
    from numba import jit
else:
    jit = lambda **kwargs: lambda *args: args[0]
