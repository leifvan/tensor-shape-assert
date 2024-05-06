from typing import Callable
import warnings
from functools import wraps
import inspect

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

class MissingOutputError(ValueError):
    pass

class NoVariableContextExistsError(RuntimeError):
    pass

_current_variables_stack = []

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

class ShapeDescriptor(tuple):
    pass

def str_to_shape_descriptor(s: str):
    s = s.replace(",", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    result = []
    
    for i in s.split(" "):
        if i == "...":
            result.append(Ellipsis)
        elif len(i) > 0:
            try:
                result.append(int(i))
            except ValueError:
                result.append(i)
                
    return ShapeDescriptor(result)


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

def check_tensor_shapes(
        variable_assertions: list[Callable] = None,
        verbose: bool = False,
        ignore_args: bool = False
):
    def wrapper_factory(fn=None, *args, **kwargs):
        if fn is None or len(args) > 0 or len(kwargs) > 0:
            raise TypeError(
                "Invalid arguments for check_tensor_shapes. Maybe you forgot "
                "brackets after the decorator?"
            )

        @wraps(fn)
        def check_wrapper(*args, **kwargs):

            if verbose:
                print("-"*30)
                print("Checking", fn.__name__)
            
            # get function signature
            signature = inspect.signature(fn)

            # bind parameters (maps parameter names to values)
            bindings = signature.bind(*args, **kwargs)
            bindings.apply_defaults()
            bound_arguments = dict(bindings.arguments)

            # collect type hints
            expected_shapes_dict = {
                k: p.annotation
                for k, p in signature.parameters.items()
                if isinstance(p.annotation, ShapeDescriptor) and k in bound_arguments
            }

            sorted_keys = list(expected_shapes_dict)

            if verbose:
                print(f"Collected {len(sorted_keys)} input shapes (and expected shapes)")
                for k in sorted_keys:
                    print(" ->", k, bound_arguments[k].shape, expected_shapes_dict[k])

            # check input shapes
            try:
                variables = assert_shapes(
                    actual_shapes=tuple(bound_arguments[k].shape for k in sorted_keys),
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

            # store variables in global stack
            _current_variables_stack.append(dict(variables)) # copy dict

            # call function
            output = fn(*args, **kwargs)

            # remove variables again
            _current_variables_stack.pop()

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
                        raise MissingOutputError(
                            f"Return annotation of function {fn.__name__} is a tuple, "
                            f"but the function returned a single tensor."
                        )

                elif getattr(fn.__annotations__["return"], "__origin__", None) == tuple: # expect a tuple of annotations
                    output_annotations = fn.__annotations__["return"].__args__

                    if len(output) < len(output_annotations):
                        raise MissingOutputError(
                            f"Expected a tuple of length {len(output_annotations)} as output of "
                            f"function {fn.__name__}, but only {len(output)} elements are returned."
                        )

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

def get_shape_variables(names: str) -> tuple[int, ...]:
    """
    Returns the inferred values of the tensor shape variables of the innermost
    function wrapped with check_tensor_shapes.

    :param names: A shape-descriptor string. See ``ShapedTensor`` for details.
    :return: A tuple of integers representing the inferred values of the variables
        given in ``names``.
    """

    if len(_current_variables_stack) == 0:
        raise NoVariableContextExistsError(
            "get_shape_variables was called without any check_tensor_shapes "
            "wrapped function in the call stack. No variables can be retrieved "
            "here."
        )
    var_names = str_to_shape_descriptor(names)
    values = tuple(_current_variables_stack[-1].get(name, None) for name in var_names)
    if len(values) == 1:
        return values[0]
    return values
