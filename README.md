# tensor-shape-assert

## Usage
Decorate functions with ``@check_tensor_shapes()`` and any parameter with a type hint of type ``ShapedTensor[<desc>]`` will be dynamically checked for the correct shape. A shape descriptor is a string of space-separated length descriptions for each dimension, where

* sizes can be defined explicitly as an integer, e.g. ``"5 3"`` (only tensors of shape ``(5, 3)`` are valid)
* ``*`` can be used as a wildcard, allowing any size, e.g. ``"* 5"`` (any 2D tensor with length 5 in the second dimension is valid)
* sizes may be given as a variable, e.g. ``"b 3"`` (any 2D tensor with length 3 in the second dimension is valid)
* an arbitrary length of batch dimensions can be defined, e.g. ``"... k 3"`` (an n-dimensional tensor (n >= 2) with length 3 in the last dimension)
* the batch dimension(s) can also be named to be matched across annotations, e.g. ``"...B n 4"`` (an n-dimensional tensor (n >= 2) with length 4 in the last dimension)
* variables can have arbitrary names, as long as they are not interpretable by the other rules, e.g. ``"my_first_dimension 123_456_test 2"`` (any 3D tensor with length 2 in the third dimension).

If multiple parameters are annotated with the same variable, the shapes must have the same length along that dimension, i.e. if a tensor ``x`` has annotation ``"a b 3"`` and another tensor ``y`` has annotation ``"b 2"``, then ``x.shape[1]`` must be equal to ``y.shape[0]``.

The return value can also be annotated in the same way. Additionally, the the annotations can be arbitrarily nested in tuples or lists. Optional ``ShapedTensor`` parameters must be explicitly annotated as a union with the ``NoneType`` (see examples below).

There are also convenience functions that access the current states of the shape variables inside the wrapped function. You can use ``get_shape_variables(<desc>)`` to retrieve a tuple of variable variables states directly, for example if you are inside a function where a tensor was annotated as ``x: ShapedTensor["a 3 b"]``, you can access the values of `a` and `b` as ``a, b = get_shape_variables("a b")``. You can even go one step further and do a check tensors inside the wrapped function directly with ``assert_shape_here(x, <desc>)``, which will run a check on the object or shape ``x`` given the descriptor and add previously unseen variables in the descriptor to the state inside the wrapped function. This way you can check the output of the function against tensor shapes that only appear in the body of the function.

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
