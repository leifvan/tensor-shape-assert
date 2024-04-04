# tensor-shape-assert

## usage
Decorate functions with ``@check_tensor_shapes()`` and any parameter with a type hint of type ``ShapedTensor[<desc>]`` will be dynamically checked for the correct shape. A shape descriptor is a string of space-separated length descriptions for each dimension, where

* sizes can be defined explicitly as an integer, e.g. ``"5 3"`` (only tensors of shape ``(5, 3)`` are valid)
* ``*`` can be used as a wildcard, allowing any size, e.g. ``"* 5"`` (any 2D tensor with length 5 in the second dimension is valid)
* some sizes may be given as a variable, e.g. ``"b 3"`` (any 2D tensor with length 3 in the second dimension is valid)
* an arbitrary length of batch dimensions can be defined, e.g. ``"... k 3"`` (an n-dimensional tensor (n >= 2) with length 3 in the last dimension)
* the batch dimension(s) can also be named to be matched across annotations, e.g. ``"...B n 4"`` (an n-dimensional tensor (n >= 2) with length 4 in the last dimension)
* variables can have arbitrary names, as long as they are not interpretable by the other rules, e.g. ``"my_first_dimension 123_456_test 2"`` (any 3D tensor with length 2 in the third dimension).

If multiple parameters are annotated with the same variable, the shapes must have the same length along that dimension, i.e. if a tensor ``x`` has annotation ``"a b 3"`` and another tensor ``y`` has annotation ``"b 2"``, then ``x.shape[1]`` must be equal to ``y.shape[0]``.

The return value can also be annotated in the same way. Additionally, the return type can be a tuple containing ``ShapedTensor`` annotations.

```python
from .tensor_shape_assert import check_tensor_shapes, ShapedTensor

@check_tensor_shapes()
def my_func(x: ShapedTensor["a b 3"], y: ShapedTensor["b 2"]) -> ShapedTensor["a"]:
    z = x[:, :, :2] + y[None]
    return (z[:, :, 0] * z[:, :, 1]).sum(dim=1)
```