from functools import partial
import weakref
from typing import Any, Iterable, Literal, NamedTuple, get_args, get_origin, TYPE_CHECKING, Dict, Callable
from array_api_compat import array_namespace
from typing_extensions import Annotated, TypeVar, TypeAliasType, LiteralString, TypeAlias

# try to import array API protocol for type annotations
from array_api._2024_12 import Array as ArrayProtocol

from .errors import (
    MalformedDescriptorError,
    UnionTypeUnsupportedError
)
from .descriptor import (
    clean_up_descriptor
)
from .utils import (
    check_if_dtype_matches
)

VariablesType = dict[str, tuple[int] | int]

# manage labels

# define str subclasses to identify shape descriptors

# _NAME_TO_KIND = {
#     'bool': 'bool',
#     'int': 'signed integer',
#     'uint': 'unsigned integer',
#     'integral': 'integral', # (int + uint)
#     'float': 'real floating',
#     'complex': 'complex floating',
#     'numeric': 'numeric' # (everything except bool)
# }
# _DTYPE_SIZES = (8, 16, 32, 64, 128, None)


class ShapeDescriptor(type):

    _label_constraint_fns: dict[str, Callable[[ArrayProtocol], bool] | None] = dict()

    @classmethod
    def register_label(
            cls,
            label: str,
            constraint_fn: Callable[[ArrayProtocol], bool] | None = None
    ):
        cls._label_constraint_fns[label] = constraint_fn

    @classmethod
    def is_registered_label(cls, label: str) -> bool:
        return label in cls._label_constraint_fns
    
    @classmethod
    def get_label_constraint_fn(cls, label: str) -> Callable[[ArrayProtocol], bool] | None:
        return cls._label_constraint_fns[label]

    @classmethod
    def filter_for_constrained_labels(cls, labels: Iterable[str]) -> frozenset[str]:
        return frozenset([label for label in labels if cls._label_constraint_fns[label] is not None])
    
    @classmethod
    def filter_for_unconstrained_labels(cls, labels: Iterable[str]) -> frozenset[str]:
        return frozenset([label for label in labels if cls._label_constraint_fns[label] is None])
    
    @classmethod
    def find_labels_in_items(cls, items: list[str]) -> list[str]:
        found_labels = []
        for item in items:
            if item in cls._label_constraint_fns:
                found_labels.append(item)
        return found_labels

    def __new__(cls, s: str):
        return type.__new__(cls, str(s), tuple(), dict())

    def __init__(self, s: str) -> None:

        # process string

        s = clean_up_descriptor(s)

        # look for labels in string (use split to ensure full words)

        self.labels: frozenset[str] = frozenset(self.find_labels_in_items(s.split(" ")))

        # remove found labels

        s = " ".join([part for part in s.split(" ") if part not in self.labels])

        self.s = clean_up_descriptor(s)


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

# register default labels

def register_label(label: str, constraint_fn: Callable[[ArrayProtocol], bool] | None = None):
    ShapeDescriptor.register_label(label, constraint_fn)

register_label('bool', lambda x: check_if_dtype_matches(x, kind='bool', bits=None))
register_label('int', lambda x: check_if_dtype_matches(x, kind='signed integer', bits=None))
register_label('int8', lambda x: check_if_dtype_matches(x, kind='signed integer', bits=8))
register_label('int16', lambda x: check_if_dtype_matches(x, kind='signed integer', bits=16))
register_label('int32', lambda x: check_if_dtype_matches(x, kind='signed integer', bits=32))
register_label('int64', lambda x: check_if_dtype_matches(x, kind='signed integer', bits=64))
register_label('uint', lambda x: check_if_dtype_matches(x, kind='unsigned integer', bits=None))
register_label('uint8', lambda x: check_if_dtype_matches(x, kind='unsigned integer', bits=8))
register_label('uint16', lambda x: check_if_dtype_matches(x, kind='unsigned integer', bits=16))
register_label('uint32', lambda x: check_if_dtype_matches(x, kind='unsigned integer', bits=32))
register_label('uint64', lambda x: check_if_dtype_matches(x, kind='unsigned integer', bits=64))
register_label('integral', lambda x: check_if_dtype_matches(x, kind='integral', bits=None))
register_label('float', lambda x: check_if_dtype_matches(x, kind='real floating', bits=None))
register_label('float16', lambda x: check_if_dtype_matches(x, kind='real floating', bits=16))
register_label('float32', lambda x: check_if_dtype_matches(x, kind='real floating', bits=32))
register_label('float64', lambda x: check_if_dtype_matches(x, kind='real floating', bits=64))
register_label('complex', lambda x: check_if_dtype_matches(x, kind='complex floating', bits=None))
register_label('complex64', lambda x: check_if_dtype_matches(x, kind='complex floating', bits=64))
register_label('complex128', lambda x: check_if_dtype_matches(x, kind='complex floating', bits=128))
register_label('numeric', lambda x: check_if_dtype_matches(x, kind='numeric', bits=None))




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
        # check if it is a tuple annotation
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError(
                    "ShapedTensor can only be parameterized with a single "
                    "shape descriptor string or a tuple of (type, shape)."
                )
            key = key[1]

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

    # TODO: this can be made more useful by using a library-specific scalar type
    ScalarTensor = TypeAliasType(
        'ScalarTensor',
        ShapedLiteral[float, Literal[""]],
        type_params=()
    )

    # torch

    try:
        import torch
        ShapedTorchLiteral = TypeAliasType(
            'ShapedTorchLiteral',
            ShapedLiteral[torch.Tensor, S],
            type_params=(S,)
        )
    except ImportError:
        pass

    # numpy

    try:
        import numpy
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