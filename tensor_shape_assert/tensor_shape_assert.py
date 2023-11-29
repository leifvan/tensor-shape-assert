import warnings
from functools import wraps
from typing import TypeVar

try:
    import torch
    TensorType = torch.Tensor
except ImportError:
    class TensorType:
        @property
        def shape(self) -> tuple[int, ...]:
            raise NotImplementedError


class IncompatibleShapeError(RuntimeError):
    def __init__(
        self,
        tensor_idx_or_name: int | str,
        actual_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
        inferred_shape: tuple[int, ...],
    ) -> None:
        if isinstance(tensor_idx_or_name, int):
            tensor_repr = tensor_idx_or_name
        else:
            tensor_repr = f"'{tensor_idx_or_name}'"
            
        super().__init__(
            f"Tensor {tensor_repr} has mismatching shape: "
            f"Expected {inferred_shape} (inferred from {expected_shape}), "
            f"but was {actual_shape}."
        )
        
        self.tensor_idx = tensor_idx_or_name
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape
        self.inferred_shape = inferred_shape


def do_shapes_match(a: tuple[int | str, ...], b: tuple[int | str, ...]):
    if len(a) != len(b):
        return False
    return all((i == j) or (i == "*") or (j == "*") for i, j in zip(a, b))


def assert_shapes(
    actual_shapes: tuple[tuple[int, ...]],
    expected_shapes: tuple[tuple[int | str, ...]],
    variables: dict[str, int] = None,
):
    variables = dict() if variables is None else variables
    for t_idx, (actual_shape, expected_shape) in enumerate(
        zip(actual_shapes, expected_shapes)
    ):
        inferred_shape = []

        if len(actual_shape) != len(expected_shape):
            raise IncompatibleShapeError(
                tensor_idx_or_name=t_idx,
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                inferred_shape=expected_shape,
            )

        for a_dim, e_dim in zip(actual_shape, expected_shape):
            if not isinstance(e_dim, int) and e_dim != "*":
                variables[e_dim] = variables.get(e_dim, a_dim)
                e_dim = variables[e_dim]
            inferred_shape.append(e_dim)

        inferred_shape = tuple(inferred_shape)

        if not do_shapes_match(actual_shape, inferred_shape):
            raise IncompatibleShapeError(
                tensor_idx_or_name=t_idx,
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                inferred_shape=inferred_shape,
            )

    return variables


def str_to_shape_descriptor(s: str):
    s = s.replace(",", "").replace("(", "").replace(")", "")
    result = []
    for i in s.split(" "):
        try:
            result.append(int(i))
        except ValueError:
            result.append(i)
    return tuple(result)


class ShapedTensor(TensorType):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "ShapedTensor instances are only meant for annotation. "
            "Instantiating is not allowed."
        )

    def __class_getitem__(cls, key):
        return str_to_shape_descriptor(key)


def check_tensor_shapes(fn):
    @wraps(fn)
    def check_wrapper(*args, **kwargs):
        if len(args) > 0:
            warnings.warn(RuntimeWarning(
                "Tensor shape checking currently does not support positional "
                "parameters. Pass all parameters with keywords instead. "
                "You can ignore this message if you decorate a method, as "
                "'self' is a positional parameter."
            ))

        # collect type hints
        shapes_dict = {
            k: v
            for k, v in fn.__annotations__.items()
            if isinstance(v, tuple) and k in kwargs
        }
        sorted_keys = list(shapes_dict)

        # check input shapes
        try:
            variables = assert_shapes(
                actual_shapes=tuple(kwargs[k].shape for k in sorted_keys),
                expected_shapes=tuple(shapes_dict[k] for k in sorted_keys),
            )
        except IncompatibleShapeError as e:
            raise IncompatibleShapeError(
                tensor_idx_or_name=sorted_keys[e.tensor_idx],
                actual_shape=e.actual_shape,
                expected_shape=e.expected_shape,
                inferred_shape=e.inferred_shape
            )

        # call function
        output = fn(*args, **kwargs)

        # check output type if annotated
        if "return" in fn.__annotations__:
            # check output shapes
            if isinstance(output, TensorType): # expect a single annotation
                output_annotation = fn.__annotations__["return"]
                
                # check if annotation is shape
                if isinstance(output_annotation, tuple):
                    assert_shapes(
                        actual_shapes=(output.shape,),
                        expected_shapes=(output_annotation,),
                        variables=variables,
                    )
                elif hasattr(fn.__annotations__["return"], "__args__") and any(
                    isinstance(a, tuple) for a in fn.__annotations__["return"].__args__
                ):
                    # assure that no tuple of tensors is annotated here
                    raise RuntimeError(
                        "Function return annotation is a tuple of at least one "
                        "shape annotation, but the function returned a single "
                        "tensor."
                    )
            else: # expect a tuple of annotations
                output_annotations = fn.__annotations__["return"].__args__
                valid_idxs = [i for i, a in enumerate(output_annotations)
                              if isinstance(a, tuple)]
                assert_shapes(
                    actual_shapes=tuple(output[i].shape for i in valid_idxs),
                    expected_shapes=tuple(output_annotations[i] for i in valid_idxs),
                    variables=variables,
                )

        return output
    return check_wrapper
