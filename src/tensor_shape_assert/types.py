import types
import inspect
from typing import Any, Callable, Literal, get_args, get_origin, TYPE_CHECKING
from typing_extensions import Annotated, TypeVar, TypeAliasType, LiteralString, TypeAlias
import warnings

# try to import array API protocol for type annotations
from array_api._2024_12 import Array as ArrayProtocol

from .utils import check_if_dtype_matches
from .errors import (
    TensorShapeAssertError,
    MalformedDescriptorError,
    UnionTypeUnsupportedError
)
from .descriptor import (
    descriptor_to_variables,
    split_to_descriptor_items,
    clean_up_descriptor
)

# define str subclasses to identify shape descriptors

_NAME_TO_KIND = {
    'bool': 'bool',
    'int': 'signed integer',
    'uint': 'unsigned integer',
    'integral': 'integral', # (int + uint)
    'float': 'real floating',
    'complex': 'complex floating',
    'numeric': 'numeric' # (everything except bool)
}
_DTYPE_SIZES = (8, 16, 32, 64, 128, None)

def find_dtype_in_items(items):
    # for any combination of dtype and size
    for dtype_size in (_DTYPE_SIZES):
            for dtype_name, kind in _NAME_TO_KIND.items():
                if dtype_size is None:
                    dtype_str = f"{dtype_name}"
                else:
                    dtype_str = f"{dtype_name}{dtype_size}"

                if dtype_str in items:
                    return (kind, dtype_size), dtype_str
    return None, ""

class ShapeDescriptor(type):
    def __new__(cls, s: str):
        return type.__new__(cls, str(s), tuple(), dict())

    def __init__(self, s: str) -> None:
        
        self.dtype = None
        self.device = None  # kept for compatibility (for now)

        # process string

        s = clean_up_descriptor(s)

        # look for dtype clues in string (use split to ensure full words)

        self.dtype, dtype_str = find_dtype_in_items(s.split(" "))

        # remove found dtype clue

        self.s = clean_up_descriptor(s.replace(dtype_str, ""))

        # check for additional descriptors

        if find_dtype_in_items(self.s.split(" "))[0] is not None:
            raise MalformedDescriptorError(
                f"Multiple dtype descriptors found in shape descriptor '{s}'. "
                f"Only a single dtype descriptor is allowed."
            )


    def __or__(self, value: Any): # type: ignore
        if value is not None:
            raise UnionTypeUnsupportedError(
                f"Union with '{value}' is not allowed as an annotation. "
                f"Currently it is only supported to use 'None' as the other "
                f"union type."
            )
        return OptionalShapeDescriptor(self.s)
    
    def __str__(self) -> str:
        return self.s
    
class OptionalShapeDescriptor(ShapeDescriptor):
    pass

class ShapedTensor(ArrayProtocol):
    """
    A helper class that allows to annotate a string that describes the shape
    of the annotated object and is then considered by the
    ``check_tensor_shapes`` wrapper. Use the generics syntax
    ``ShapedTensor[<desc>]`` to annotate the objects, where ``<desc>`` is a
    string which contains a whitespace-separated list of shape descriptions
    for each of the dimensions. More particularly these dimension descriptors
    may be
    * an integer, to test against a fixed size,
    * ``'*'`` to denote an arbitrary size along that dimension,
    * ``'...'`` followed by an optional name to denote an arbitrary number of
    dimensions (only allowed once per shape),
    * a string that does not fulfill any of the  rules above, which is then
    interpreted as a variable and checked for equality across all annotated 
    arguments and return values of the function.

    Note that most punctuation is replaced with whitespaces, so for example
    it is also possible use a comma-separated list like
    ``ShapedTensor["a,b,c"]`` instead of ``ShapedTensor["a b c"]``.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ShapedTensor instances are only meant for annotation. "
            "Instantiating is not allowed."
        )
    
    def __class_getitem__(cls, key):
        # check if it is a literal
        if get_origin(key) is Literal:
            key = " ".join(get_args(key))

        return ShapeDescriptor(key)


if TYPE_CHECKING:
    T = TypeVar('T')
    S = TypeVar('S', bound=LiteralString)

    ShapedLiteral = TypeAliasType(
        'ShapedLiteral',
        Annotated[T, S],
        type_params=(T, S)
    )

    # torch

    try:
        from array_api_compat import torch
        ShapedTorchLiteral = TypeAliasType(
            'ShapedTorchLiteral',
            ShapedLiteral[torch.Tensor, S],
            type_params=(S,)
        )
    except ImportError:
        pass

    # numpy

    try:
        from array_api_compat import numpy
        ShapedNumpyLiteral = TypeAliasType(
            'ShapedNumpyLiteral',
            ShapedLiteral[numpy.ndarray, S],
            type_params=(S,)
        )
    except ImportError:
        pass

else:
    ShapedLiteral = ShapedTensor
    ShapedTorchLiteral = ShapedTensor
    ShapedNumpyLiteral = ShapedTensor

    # aliases
    ScalarTensor: TypeAlias = ShapedTensor[""] 