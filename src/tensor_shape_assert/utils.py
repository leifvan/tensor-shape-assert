from __future__ import annotations

import weakref
from typing import Generic, TypeVar

from array_api_compat import array_namespace
from array_api._2024_12 import Array as ArrayProtocol
from .errors import TensorShapeAssertError

def optional_to_int(s: str):
    try:
        return int(s)
    except ValueError:
        return s


def check_if_dtype_matches(obj, kind, bits):
    api = array_namespace(obj)
    
    # dtype does not match
    if not api.isdtype(obj.dtype, kind):
        return False

    # bits not given -> accept
    if bits is None:
        return True

    # check bits via info function
    if api.isdtype(obj.dtype, "integral"):
        return api.iinfo(obj.dtype).bits == bits
    elif api.isdtype(obj.dtype, "real floating"):
        return api.finfo(obj.dtype).bits == bits
    elif api.isdtype(obj.dtype, "complex floating"):
        return 2 * api.finfo(obj.dtype).bits == bits
    else:
        raise TensorShapeAssertError(
            f"Dtype '{obj.dtype}' does not support bit checking."
        )

def is_typing_namedtuple_instance(x) -> bool:
    t = type(x)
    return (
        isinstance(x, tuple)
        and hasattr(t, "_fields")
        and isinstance(getattr(t, "_fields", None), tuple)
        and hasattr(t, "__annotations__")  # typing.NamedTuple gives annotations
    )


V = TypeVar("V")

class ArrayIdentityMap(Generic[V]):
    def __init__(self) -> None:
        self._values: dict[int, V] = {}
        self._finalizers: dict[int, weakref.finalize] = {}

    def make_key(self, tensor: ArrayProtocol) -> int:
        return hash((id(tensor), tensor.shape, tensor.dtype))

    def __setitem__(self, tensor: ArrayProtocol, value: V) -> None:
        key = self.make_key(tensor)

        # Remove old finalizer if we overwrite the same object entry
        old_finalizer = self._finalizers.pop(key, None)
        if old_finalizer is not None:
            old_finalizer.detach()

        self._values[key] = value
        self._finalizers[key] = weakref.finalize(
            tensor,
            self._remove,
            key,
        )

    def __getitem__(self, tensor: ArrayProtocol) -> V:
        return self._values[self.make_key(tensor)]

    def get(self, tensor: ArrayProtocol, default: V | None = None) -> V | None:
        return self._values.get(self.make_key(tensor), default)

    def __contains__(self, tensor: ArrayProtocol) -> bool:
        return self.make_key(tensor) in self._values

    def pop(self, tensor: ArrayProtocol, default: V | None = None) -> V | None:
        key = self.make_key(tensor)
        finalizer = self._finalizers.pop(key, None)
        if finalizer is not None:
            finalizer.detach()
        return self._values.pop(key, default)

    def _remove(self, key: int) -> None:
        self._values.pop(key, None)
        self._finalizers.pop(key, None)