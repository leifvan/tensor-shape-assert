from functools import wraps
import types
import inspect
from typing import Any, Callable, ForwardRef, Literal, TypeVar
import warnings

from .utils import TensorShapeAssertError
from .descriptor import descriptor_to_variables, split_to_descriptor_items, clean_up_descriptor

# define errors

class NoVariableContextExistsError(TensorShapeAssertError):
    pass

class AnnotationMatchingError(TensorShapeAssertError):
    pass

class VariableConstraintError(TensorShapeAssertError):
    pass

class DtypeConstraintError(TensorShapeAssertError):
    pass

class DeviceConstraintError(TensorShapeAssertError):
    pass


# try importing torch for type hints

try:
    import torch
    TensorType = torch.Tensor
except ImportError:
    class TensorType:
        @property
        def shape(self) -> tuple[int, ...]:
            raise NotImplementedError
        
        @property
        def dtype(self) -> Any:
            raise NotImplementedError
        
        @property
        def device(self) -> Any:
            raise NotImplementedError

# define str subclasses to identify shape descriptors

class ShapeDescriptor(type):
    def __new__(cls, s: str):
        return type.__new__(cls, str(s), tuple(), dict())

    def __init__(self, s: str | tuple[str, Any] | tuple[str, Any, Any]) -> None:
        self.dtype = self.device = None
        
        if isinstance(s, tuple):
            if len(s) == 3:
                self.s, self.dtype = s[:2]
                self.device = torch.device(s[2])
            elif len(s) == 2:
                self.s, self.dtype = s
            else:
                raise TensorShapeAssertError(
                    f"Incorrect shape descriptor '{s}'. Has to be a string or a tuple "
                    f"(string, dtype) or (string, dtype, device)."
                )
        else:
            self.s = s

    def __or__(self, value: Any) -> types.GenericAlias:
        if value is not None:
            raise TensorShapeAssertError(
                f"Union with '{value}' is not allowed as an annotation. Currently it is "
                f"only supported to use 'None' as the other union type."
            )
        return OptionalShapeDescriptor(self.s)
    
    def __str__(self) -> str:
        return self.s
    
class OptionalShapeDescriptor(ShapeDescriptor):
    pass

class ShapedTensor(TensorType):
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
        return ShapeDescriptor(key)
    

def unroll_iterable_annotation(annotation, obj):
    if isinstance(annotation, (ShapeDescriptor, OptionalShapeDescriptor)):
        yield annotation, obj

    elif isinstance(annotation, types.GenericAlias):
        # try to infer how annotation maps to iterable 
        sub_annotations = None

        if annotation.__origin__ == tuple:
            sub_annotations = annotation.__args__

            # check if its a arbitrary length annotation
            if len(sub_annotations) == 2 and sub_annotations[1] == Ellipsis:
                sub_annotations = [sub_annotations[0]] * len(obj)

        elif annotation.__origin__ == list:
            if len(annotation.__args__) != 1:
                raise AnnotationMatchingError(
                    f"Expected a single argument for list annotation, but got "
                    f"{annotation.__args__}."
                )
            sub_annotations = [annotation.__args__[0]] * len(obj)

        if sub_annotations is not None:
            
            if len(sub_annotations) != len(obj):
                raise AnnotationMatchingError(
                    f"Number of expected annotated elements in iterable "
                    f"type does not match actual length. Expected "
                    f"{len(sub_annotations)} items, but got {len(obj)}."
                )
            
            for sub_ann, sub_obj in zip(sub_annotations, obj):
                yield from unroll_iterable_annotation(sub_ann, sub_obj)

    elif isinstance(annotation, types.UnionType):
        for arg in annotation.__args__:
            if isinstance(arg, types.GenericAlias):
                warnings.warn(RuntimeWarning(
                    "You used a union type in a function to be checked by "
                    "tensor_shape_assert. check_tensor_shapes currently does "
                    "not check iterables of tensors in union types. For "
                    "example, the output of \n\n"
                    "    def test() -> tuple[ShapedTensor['1']] | None:\n\n"
                    "will be ignored, except for the case where a single "
                    "ShapedTensor is found in a union. For example\n\n"
                    "    def test() -> ShapedTensor['1'] | None:\n\n"
                    "is allowed and will be checked as expected."
                ))


def check_iterable(annotation, obj, variables):
    for descriptor, obj in unroll_iterable_annotation(annotation, obj):

        # skip if its optional and obj is None
        if isinstance(descriptor, OptionalShapeDescriptor) and obj is None:
            continue

        # check shape
        variables = descriptor_to_variables(
            shape_descriptor=str(descriptor),
            shape=obj.shape,
            variables=variables
        )

        # optionally check dtype
        if descriptor.dtype is not None:
            if obj.dtype != descriptor.dtype:
                raise DtypeConstraintError(
                    f"Dtype '{obj.dtype}' does not match the annotated dtype "
                    f"'{descriptor.dtype}'."
                )
            
        # optionally check device
        if descriptor.device is not None:
            if obj.device != descriptor.device:
                raise DeviceConstraintError(
                    f"Device '{obj.device}' does not match the annotated device "
                    f"'{descriptor.device}'."
                )
    
    return variables

# define a module level stack for currently declared variables
_current_variables_stack = []

# module level check mode
CheckMode = Literal["always", "once", "never"]
_global_check_mode: CheckMode = "always"
_checked_functions: set = set()

def set_global_check_mode(mode: CheckMode):
    global _global_check_mode
    _global_check_mode = mode


def run_expression_constraint(
        expression: str,
        variables: dict[str, int]
) -> bool:
    if "=" not in expression:
        assert False
    if "==" not in expression:
        expression = expression.replace("=", "==")

    exec_globals = {'__builtins__': {}, **variables}
    return eval(expression, exec_globals)

def check_constraints(
        constraints: list[Callable[[dict[str, int]], bool] | str],
        variables: dict[str, int],
        skip_on_error: bool
):
    for i, constraint_fn in enumerate(constraints):
        try:
            if isinstance(constraint_fn, str):
                passed = run_expression_constraint(constraint_fn, variables)
                name = f"{i} [{constraint_fn}]"
            else:
                passed = constraint_fn(variables)
                name = f"{i}"
        except Exception:
            if skip_on_error:
                passed = True
            else:
                raise

        if not passed:
            raise VariableConstraintError(
                f"Constraint {name} was not fulfilled for variable "
                f"assignments {variables}."
            )


def check_tensor_shapes(
        constraints: list[str | Callable[[dict[str, int]], bool]] = None,
        ints_to_variables: bool = True,
        experimental_enable_autogen_constraints: bool = False,
        check_mode: CheckMode | None = None,
        *args, **kwargs
):
    """
    Enables tensor checking for the decorated function.

    Parameters
    ----------

    constraints : list, optional
        A list of string expressions or callables that are
        run for the variable assignments before and after the wrapped function
        is called. If callables are given, they receive the variable assignments
        as a dictionary and are expected to return ``True`` if the constraint
        is fullfilled, ``False`` otherwise.
    
    ints_to_variables : bool, optional
        If ``True`` (default), all function parameters of type ``int`` will be added to
        the list of shape variables and can be used in a shape descriptor.

    experimental_enable_autogen_constraints : bool, optional
        If ``True``, all variable names will be compiled as constraints. This
        comes with two caveats: First of all, as names are split by whitespaces,
        naming the variables as expressions does not support any whitespaces.
        Additionally, variable names must follow the Python variable naming
        conventions.
    """

    if len(args) > 0 or len(kwargs) > 0:
        raise TensorShapeAssertError(
            "Invalid arguments for check_tensor_shapes. Maybe you forgot "
            "brackets after the decorator?"
        )
    
    if constraints is None:
        constraints = []
    
    def _make_check_tensor_shapes_wrapper(fn=None, *args, **kwargs):
        if fn is None or len(args) > 0 or len(kwargs) > 0:
            raise TensorShapeAssertError(
                "Invalid arguments for check_tensor_shapes. Maybe you forgot "
                "brackets after the decorator?"
            )

        @wraps(fn)
        def _check_tensor_shapes_wrapper(*args, **kwargs):
            
            # get function signature
            
            signature = inspect.signature(fn)

            # shortcut function if check mode is different to 'always'

            _check_mode = _global_check_mode if check_mode is None else check_mode

            if  (
                    (_check_mode == 'once'
                    and signature in _checked_functions)
                    or _check_mode == 'never'
                ):
                return fn(*args, **kwargs)
            
            _checked_functions.add(signature)

            # bind parameters
            
            bindings = signature.bind(*args, **kwargs)
            bindings.apply_defaults()
            bound_arguments = dict(bindings.arguments)

            # check input type hints

            if ints_to_variables:
                variables = {k: v for k, v in bound_arguments.items() if type(v) is int}
            else:
                variables = dict()

            for key, parameter in signature.parameters.items():
                try:
                    variables = check_iterable(
                        annotation=parameter.annotation,
                        obj=bound_arguments[key],
                        variables=variables
                    )
                except TensorShapeAssertError as e:
                    # wrap exception to provide location info (input)
                    raise TensorShapeAssertError(
                        f"Shape assertion failed during check of input "
                        f"parameter '{key}' of '{fn.__name__}': {e}"
                    )
                
            # check input variable constraints (allow errors for now)
            check_constraints(constraints, variables, skip_on_error=True)

            if experimental_enable_autogen_constraints:
                for i, (k, v) in enumerate(variables.items()):
                    temp_variables = dict(variables)
                    temp_replace_name = f"__temp_var_{i}"
                    temp_replace_val = temp_variables[k]
                    del temp_variables[k]
                    temp_variables[temp_replace_name] = temp_replace_val
                    temp_constraint = f"{temp_replace_name} == {k}"
                    check_constraints([temp_constraint], temp_variables, skip_on_error=True)

            # store variables in global stack
            _current_variables_stack.append(variables)

            # call function
            # protect function call to gracefully handle variables stack

            try:
                return_value = fn(*args, **kwargs)
            except Exception:
                _current_variables_stack.pop()
                raise  # reraise
            
            # check outputs

            try:
                check_iterable(
                    annotation=signature.return_annotation, 
                    obj=return_value,
                    variables=variables
                )
            except TensorShapeAssertError:
                # wrap exception to provide location info (output)
                raise TensorShapeAssertError(
                    f"Shape assertion failed during check output of "
                    f"'{fn.__name__}': {e}"
                )
            finally:
                # check output variable constraints (this time strictly)
                check_constraints(constraints, variables, skip_on_error=False)

                if experimental_enable_autogen_constraints:
                    for i, (k, v) in enumerate(variables.items()):
                        temp_variables = dict(variables)
                        temp_replace_name = f"__temp_var_{i}"
                        temp_replace_val = temp_variables[k]
                        # del temp_variables[k]
                        temp_variables[temp_replace_name] = temp_replace_val
                        temp_constraint = f"{temp_replace_name} == {k}"
                        check_constraints([temp_constraint], temp_variables, skip_on_error=False)

                # remove vars from stack
                _current_variables_stack.pop()
            
            # return

            return return_value

        return _check_tensor_shapes_wrapper

    return _make_check_tensor_shapes_wrapper

def check_if_context_is_available():
    if len(_current_variables_stack) == 0:
        raise NoVariableContextExistsError(
            "get_shape_variables was called without any check_tensor_shapes "
            "wrapped function in the call stack. No variables can be retrieved "
            "here."
        )

def get_shape_variables(names: str) -> tuple[int, ...]:
    """
    Returns the inferred values of the tensor shape variables of the innermost
    function wrapped with check_tensor_shapes.

    Parameters
    ----------
    
    names : str
        A shape descriptor string. See ``ShapedTensor`` for details.
    
    Returns
    -------
    tuple[int]
        A tuple of integers representing the inferred values of the variables
        given in ``names``.
    """
    check_if_context_is_available()
    
    front, back, mdd = split_to_descriptor_items(names)
    if mdd is None:
        var_names = front
    else:
        var_names = (*front, mdd, *back)

    values = tuple(_current_variables_stack[-1].get(name, None) for name in var_names)
    if len(values) == 1:
        return values[0]
    return values

def assert_shape_here(obj_or_shape: Any, descriptor: str) -> None:
    """
    Checks if the given object or shape matches the given ``descriptor``.
    Variables in the descriptor not previously defined in the wrapped function
    will be set to the appropriate value and are used in subsequent calls to
    functions accessing the states of the shape variables, which includes
    the check of the function's output.

    Parameters
    ----------
    obj_or_shape
        Either an object with a ``.shape`` property or a shape to be checked
        against ``descriptor``.
    descriptor : str
        A shape descriptor string. See ``ShapedTensor`` for details.
    """
    check_if_context_is_available()

    try:
        shape = obj_or_shape.shape
    except Exception:
        shape = obj_or_shape

    _current_variables_stack[-1] = descriptor_to_variables(
        shape_descriptor=descriptor,
        shape=shape,
        variables=_current_variables_stack[-1]
    )
