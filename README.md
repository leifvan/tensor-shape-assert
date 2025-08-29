# tensor-shape-assert

## Usage
Decorate functions with ``@check_tensor_shapes()`` and any parameter with a type hint of type ``ShapedTensor[<desc>]`` will be dynamically checked for the correct shape. A shape descriptor is a string of space-separated length descriptions for each dimension, where

* sizes can be defined explicitly as an integer, e.g. ``"5 3"`` (only tensors of shape ``(5, 3)`` are valid)
* ``*`` can be used as a wildcard, allowing any size, e.g. ``"* 5"`` (any 2D tensor with length 5 in the second dimension is valid)
* sizes may be given as a variable, e.g. ``"b 3"`` (any 2D tensor with length 3 in the second dimension is valid)
* an arbitrary length of batch dimensions can be defined, e.g. ``"... k 3"`` (an n-dimensional tensor (n >= 2) with length 3 in the last dimension)
* the batch dimension(s) can also be named to be matched across annotations, e.g. ``"...B n 4"`` (an n-dimensional tensor (n >= 2) with length 4 in the last dimension)
* variables can have arbitrary names, as long as they are not interpretable by the other rules, e.g. ``"my_first_dimension 123_456_test 2"`` (any 3D tensor with length 2 in the third dimension). See also dtype annotations below.
* using an empty string ``""`` or the alias ``ScalarTensor`` checks for scalar tensors.

If multiple parameters are annotated with the same variable, the shapes must have the same length along that dimension, i.e. if a tensor ``x`` has annotation ``"a b 3"`` and another tensor ``y`` has annotation ``"b 2"``, then ``x.shape[1]`` must be equal to ``y.shape[0]``.

The return value can also be annotated in the same way. Additionally, the the annotations can be arbitrarily nested in tuples or lists. Optional ``ShapedTensor`` parameters must be explicitly annotated as a union with the ``NoneType`` (see examples below).

Parameters of type ``int`` are added to the list of shape variables, which allows to specify fixed shapes dynamically. This behavior can be turned off with ``@check_tensor_shapes(ints_to_variables=False)``. An example is shown below.

The dtype can be annotated by adding a dtype kind (``bool, int, uint, float, complex, numeric``) and optionally a bit size (e.g. ``uint8``) anywhere in the descriptor. Examples: ``"uint8 a b 3"``, or ``"a b 3 float"``. The names above can therefore not be used as a variable names.

There are convenience functions that access the current states of the shape variables inside the wrapped function. You can use ``get_shape_variables(<desc>)`` to retrieve a tuple of variable variables states directly, for example if you are inside a function where a tensor was annotated as ``x: ShapedTensor["a 3 b"]``, you can access the values of `a` and `b` as ``a, b = get_shape_variables("a b")``. You can even go one step further and do a check tensors inside the wrapped function directly with ``assert_shape_here(x, <desc>)``, which will run a check on the object or shape ``x`` given the descriptor and add previously unseen variables in the descriptor to the state inside the wrapped function. This way you can check the output of the function against tensor shapes that only appear in the body of the function.

## Installation

Currently, the package can only be installed directly from the repository with
```bash
pip install git+https://github.com/leifvan/tensor-shape-assert
```

## Compatibility

While the examples below are using PyTorch, tensor-shape-assert requires very minimal functionality and is compatible with any array class that has a ``shape`` method, which includes popular frameworks such as NumPy, TensorFlow, Jax and more generally frameworks that conform to the [Python array API standard](https://data-apis.org/array-api/latest/).

## Examples

Here are two examples that demonstrate how the annotation works.

### Simple example

```python
import torch
from .tensor_shape_assert import check_tensor_shapes, ShapedTensor

@check_tensor_shapes()
def my_simple_func(
        x: ShapedTensor["a b 3"],
        y: ShapedTensor["b 2"]
) -> ShapedTensor["a"]:

    z = x[:, :, :2] + y[None]
    return (z[:, :, 0] * z[:, :, 1]).sum(dim=1)
```

Calling it like this
```python
my_simple_func(torch.zeros(5, 4, 3), y=torch.zeros(4, 2)) # works
```
passes the test, because ``a=5 and b=4`` matches for both input and output annotations.

For
```python
my_simple_func(torch.zeros(5, 4, 3), y=torch.zeros(4, 3)) # fails
```
the test fails, because `y` is expected to have length 2 in the second dimension.

---
### Complex example

The complex example additionally contains tuple and optional annotations.
```python
@check_tensor_shapes()
def my_complicated_func(
        x: tuple[
            ShapedTensor["a ... 3"],
            ShapedTensor["b"] | None,
            ShapedTensor["c 2"]
        ],
        y: ShapedTensor["... c"]
) -> tuple[
    ShapedTensor["a"],
    ShapedTensor["b"] | None
]:
    x1, x2, x3 = x
    z = x1[..., 2:] # (a, ..., 1)
    r = x3[:, 0] + y # (..., c)
    f = r[None] + z # (a, ..., c)
    g = f.flatten(1).sum(dim=1) # (a,)

    if x2 is not None and x2.sum() > 0:
        return g, x2
    else:
        return g, None
```

Here are some calling examples:

```python
my_complicated_func(
    x=(
        torch.zeros(5, 4, 3),
        torch.zeros(8),
        torch.zeros(4, 2),
    ),
    y=torch.zeros(4, 4)
) # works
```
This works, because ``a=5, b=8, c=4`` and the batch dimension ``(4,)`` matches for all annotated tensors.

```python
my_complicated_func(
    x=(
        torch.zeros(5, 4, 3),
        None,
        torch.zeros(4, 2),
    ),
    y=torch.zeros(4, 4)
) # works
```
This call also passes the test, because the second item in `x` is allowed to be optional, whereas

```python
my_complicated_func(
    x=(
        torch.zeros(5, 3, 6, 3),
        torch.zeros(8),
        torch.zeros(4, 2),
    ),
    y=torch.zeros(4, 4)
) # fails
```
fails, because the batch dimension does not match between the first item in `x` (batch dim = `(3,6)`) and tensor `y` (batch dim = `(4,)`).

---
### Retrieve shape variables
You can access the shape variable values using ``get_shape_variables`` like this
```python
@check_tensor_shapes()
def my_func(x: ShapedTensor["n k 3"]):
    n, k = get_shape_variables("n k")
    print(k)
my_func(torch.zeros(10, 9))  # prints "9"
```

---
### int to variables
If ``int`` parameters are present, they can be used inside the shape descriptors:
```python
@check_tensor_shapes()
def my_func(x: ShapedTensor["n k"], k: int):
    return x.sum(dim=1)

my_func(torch.zeros(10, 2), k=2) # works
my_func(torch.zeros(10, 2), k=3) # fails
```

unless this functionality is explicitly turned off:
```python
@check_tensor_shapes(ints_to_variables=False)
def my_func(x: ShapedTensor["n k"], k: int):
    return x.sum(dim=1)

my_func(torch.zeros(10, 2), k=2) # works
my_func(torch.zeros(10, 2), k=3) # works
```

## Known bugs
* [ ] Variable stack collects int values as variables and prints them, i.e. it
 produces error messages like ``"Shape torch.Size([1, 4, 4]) does not match
 descriptor (b, 4, 4) at position 0 based on already inferred variables 
 {'b': 4, 'n': 304, 2: 2, 1: 1, 4: 4}"``
* [ ] ``get_shape_variables`` does not work if checks are disabled. This should
 be possible but give a performance warning, recommending not to use this
 feature in performance-critical applications.

## Future Improvements
These are feature that are not implemented yet, but might be added in future
releases.

* [x] dtype annotation
* ~~[ ] device annotation~~ (device definition not standardized in Python array API 2024.12, see [this section of the specifications](https://data-apis.org/array-api/2024.12/design_topics/device_support.html#device-support))
* [ ] add tests for autogenerated constraints and come up with a specific 
 syntax to enable it (or enable it by default?)
* [x] make exception messages more concise and remove currently used exception 
reraise
* [ ] improve annotation handling for method overrides in subclasses
* [ ] add tests for frameworks other than PyTorch
  * [x] numpy
  * [ ] jax
  * [ ] cupy
  * [ ] sparse
  * [ ] donnx
  * [ ] dask
* [ ] check compatibility with static type checkers
* [ ] rewrite README to give a cleaner overview over the features
* [ ] support union of shape descriptors (but this might break the current simplicity)
* [ ] benchmark speed to understand impact in tight loops
* [x] compatibility for torch.compile (or at least auto-disable check)