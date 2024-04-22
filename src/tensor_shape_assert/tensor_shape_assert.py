from typing import Callable
import warnings
from functools import wraps

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
        fn_name: str = "<unk>"
    ) -> None:
        if isinstance(tensor_idx_or_name, int):
            tensor_type = "Output"
            tensor_repr = tensor_idx_or_name
        else:
            tensor_type = "Input"
            tensor_repr = f"'{tensor_idx_or_name}'"
            
        super().__init__(
            f"{tensor_type} tensor {tensor_repr} of function '{fn_name}' has "
            f"mismatching shape: Expected {inferred_shape} (inferred from "
            f"{expected_shape}), but was {actual_shape}."
        )
        
        self.tensor_idx = tensor_idx_or_name
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape
        self.inferred_shape = inferred_shape

class VariableAssertionError(RuntimeError):
    pass

class IllegalPositionalArgumentError(ValueError):
    pass


def do_shapes_match(a: tuple[int | str, ...], b: tuple[int | str, ...]):
    if len(a) != len(b):
        return False
    return all((i == j) or (i == "*") or (j == "*") for i, j in zip(a, b))

def assert_shapes(
    actual_shapes: tuple[tuple[int, ...]],
    expected_shapes: tuple[tuple[int | str, ...]],
    variables: dict[str, int] = None,
    verbose: bool = False
):
    variables = dict() if variables is None else variables
    for t_idx, (actual_shape, expected_shape) in enumerate(
        zip(actual_shapes, expected_shapes)
    ):
        if verbose:
            print(f"shape check {t_idx}: got {actual_shape}, expect {expected_shape}")

        inferred_shape = []
        
        # # only allow ellipses as the starting dim
        if any(i == Ellipsis or (isinstance(i, str) and i.startswith("...")) for i in expected_shape[1:]):
            raise ValueError(
                f"'...' is only allowed as the first dimension, but annotated "
                f"shape was {expected_shape}."
            )

        # replace unnamed batch dimension with *
        if expected_shape[0] == Ellipsis:
            expected_shape = (*(['*'] * (len(actual_shape) - len(expected_shape) + 1)), *expected_shape[1:])

            if verbose:
                print("Replacing ellipsis with wildcards - new expected shape:", expected_shape)
        
        # replace named batch dimension with recorded size
        elif isinstance(expected_shape[0], str) and expected_shape[0].startswith("..."):
            if expected_shape[0] not in variables:
                variables[expected_shape[0]] = actual_shape[:-len(expected_shape)+1]
            
            expected_shape = (*variables[expected_shape[0]], *expected_shape[1:])

            if verbose:
                print("Replacing named batch dim with actual shape - new expected shape:", expected_shape)

        # check if number of dimensions is the same
        if len(actual_shape) != len(expected_shape):
            raise IncompatibleShapeError(
                tensor_idx_or_name=t_idx,
                actual_shape=actual_shape,
                expected_shape=expected_shape,
                inferred_shape=expected_shape,
            )

        # track variables
        for a_dim, e_dim in zip(actual_shape, expected_shape):
            if not isinstance(e_dim, int) and e_dim != "*":
                variables[e_dim] = variables.get(e_dim, a_dim)

                if verbose:
                    print(f"Get value for variable: {e_dim} = {variables[e_dim]}")

                e_dim = variables[e_dim]
            inferred_shape.append(e_dim)

        inferred_shape = tuple(inferred_shape)
        if verbose:
            print("final inferred shape:", inferred_shape, "to be checked against", actual_shape)

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
        if i == "...":
            result.append(Ellipsis)
        else:
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

def check_variable_assertions(variable_assertions: dict[str, Callable] | None, variables: dict[str, int]):
    if variable_assertions is not None:
        for var_name, assert_fn in variable_assertions.items():
            if var_name in variables:
                try:
                    assert_success = assert_fn(variables)
                except Exception as e:
                    raise RuntimeError("An exception was raised inside assert function.")
                
                if not assert_success:
                        raise VariableAssertionError(f"An assertion failed for variables {variables}")

def check_tensor_shapes(variable_assertions: list[Callable] = None, verbose: bool = False, ignore_args: bool = False):
    def wrapper_factory(fn):
        @wraps(fn)
        def check_wrapper(*args, **kwargs):

            if verbose:
                print("-"*30)
                print("Checking", fn.__name__)

            if len(args) == 1:
                warnings.warn(RuntimeWarning(
                    "Tensor shape checking currently does not support positional "
                    "parameters. Pass all parameters with keywords instead. "
                    "You can ignore this message if you decorate a method, as "
                    "'self' is a positional parameter."
                ))
            elif len(args) > 1 and not ignore_args:
                raise IllegalPositionalArgumentError(
                    "Tensor shape checking currently does not support positional "
                    "parameters. Pass all parameters with keywords instead."
                )

            # collect type hints
            expected_shapes_dict = {
                k: v
                for k, v in fn.__annotations__.items()
                if isinstance(v, tuple) and k in kwargs
            }
            sorted_keys = list(expected_shapes_dict)

            if verbose:
                print(f"Collected {len(sorted_keys)} input shapes (and expected shapes)")
                for k in sorted_keys:
                    print(" ->", k, kwargs[k].shape, expected_shapes_dict[k])

            # check input shapes
            try:
                variables = assert_shapes(
                    actual_shapes=tuple(kwargs[k].shape for k in sorted_keys),
                    expected_shapes=tuple(expected_shapes_dict[k] for k in sorted_keys),
                )
            except IncompatibleShapeError as e:
                raise IncompatibleShapeError(
                    tensor_idx_or_name=sorted_keys[e.tensor_idx],
                    actual_shape=e.actual_shape,
                    expected_shape=e.expected_shape,
                    inferred_shape=e.inferred_shape,
                    fn_name=fn.__name__
                )
            
            if verbose:
                print("Input shapes check successful.")

            # check assert functions for variable values
            check_variable_assertions(variable_assertions, variables)

            if verbose:
                print("Input variable assertions check successful.")

            # call function
            output = fn(*args, **kwargs)

            # check output type if annotated
            if "return" in fn.__annotations__:
                # check output shapes
                if isinstance(output, TensorType): # expect a single annotation
                    output_annotation = fn.__annotations__["return"]

                    if verbose:
                        print("Collected a single output annotation:", output.shape, output_annotation)
                    
                    # check if annotation is shape
                    if isinstance(output_annotation, tuple):
                        variables = assert_shapes(
                            actual_shapes=(output.shape,),
                            expected_shapes=(output_annotation,),
                            variables=variables
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
                                  if isinstance(a, tuple)]  # tuple means converted string annotation here
                    
                    if verbose:
                        print(f"Collected {len(valid_idxs)}/{len(output_annotations)} output annotations:")
                        for i in valid_idxs:
                            print(" ->", i, output[i].shape, output_annotations[i])

                    try:
                        variables = assert_shapes(
                        actual_shapes=tuple(output[i].shape for i in valid_idxs),
                        expected_shapes=tuple(output_annotations[i] for i in valid_idxs),
                        variables=variables
                    )
                    except IncompatibleShapeError as e:
                        raise IncompatibleShapeError(
                            tensor_idx_or_name=e.tensor_idx,
                            actual_shape=e.actual_shape,
                            expected_shape=e.expected_shape,
                            inferred_shape=e.inferred_shape,
                            fn_name=fn.__name__
                        )
            
            if verbose:
                print("Output shapes check successful.")
            
            # check assert functions again after receiving output
            check_variable_assertions(variable_assertions, variables)

            if verbose:
                print("Output variable assertions check successful.")
            
            return output
        return check_wrapper
    return wrapper_factory
