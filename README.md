# tensor-shape-assert

**Runtime tensor shape and dtype checking through type annotations.**

`tensor-shape-assert` validates the shapes (and optionally dtypes) of array-like objects at function call time, based on annotations you already write. Shared dimension variables are automatically inferred and matched across all annotated parameters and return values — a mismatch raises a clear error before your computation runs.

Compatible with any array library that exposes a `.shape` property, including NumPy, PyTorch, JAX, and TensorFlow.

## Features

- Runtime shape validation via `ShapedTensor["..."]` type annotations
- Shape **variables** inferred from — and matched across — multiple parameters and return values
- **Batch dimension** support with named and unnamed ellipsis tokens (`...`, `...B`)
- **Dtype annotations** (`bool`, `int8`, `float32`, `complex128`, …)
- **Optional** and **nested** annotations (tuples, lists, `NamedTuple`)
- `int` parameters automatically promoted to shape variables
- Per-function and global **check modes** (`always`, `once`, `never`) for zero-overhead production deploys
- Compatible with static type checkers (MyPy, Pyright) via `ShapedLiteral` aliases

## Installation

```bash
pip install git+https://github.com/leifvan/tensor-shape-assert
```

## Quick Start

```python
import numpy as np
from tensor_shape_assert import check_tensor_shapes, ShapedTensor

@check_tensor_shapes()
def matrix_multiply(
        x: ShapedTensor["batch m k"],
        y: ShapedTensor["batch k n"],
) -> ShapedTensor["batch m n"]:
    return x @ y

matrix_multiply(np.zeros((4, 5, 3)), np.zeros((4, 3, 7)))  # passes
matrix_multiply(np.zeros((4, 5, 3)), np.zeros((4, 2, 7)))  # raises TensorShapeAssertError
```

The decorator infers `batch=4`, `m=5`, `k=3` from `x` and checks that `y` and the return value are consistent with those values.

## Shape Descriptor Syntax

A shape descriptor is a whitespace-separated string (most punctuation is also treated as whitespace). Each token describes one dimension:

| Token | Meaning |
|-------|---------|
| `5` | Exact size 5 |
| `*` | Wildcard — any size |
| `n` | Variable — resolved and matched across all annotations that use the same name |
| `...` | Zero or more batch dimensions (may appear at most once) |
| `...B` | Named batch dimensions — must match across annotations sharing the same name `B` |
| `""` / `ScalarTensor` | Scalar (0-dimensional) tensor |

Dtype tokens can appear anywhere in the descriptor alongside dimension tokens (see [Dtype Annotations](#dtype-annotations)).

## Core Concepts

### Variables

When two parameters share a variable name, their sizes along that dimension must agree:

```python
@check_tensor_shapes()
def add(x: ShapedTensor["n k"], y: ShapedTensor["n k"]) -> ShapedTensor["n k"]:
    return x + y
```

Variable names can be any identifier not reserved by other rules (integers, `*`, `...`, dtype tokens).

### Integers as Shape Variables

`int` parameters are automatically promoted to shape variables, enabling dynamic shape constraints:

```python
@check_tensor_shapes()
def take_k(x: ShapedTensor["n k"], k: int) -> ShapedTensor["n k"]:
    return x[:, :k]

take_k(np.zeros((10, 4)), k=4)  # passes — k=4 matches x.shape[1]
take_k(np.zeros((10, 4)), k=3)  # raises TensorShapeAssertError
```

Disable this behaviour with `@check_tensor_shapes(ints_to_variables=False)`.

### Batch Dimensions

Use `...` for an arbitrary number of leading dimensions:

```python
@check_tensor_shapes()
def normalize(x: ShapedTensor["... d"]) -> ShapedTensor["... d"]:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
```

Use a named batch dimension (`...B`) to enforce that multiple parameters share the same batch shape:

```python
@check_tensor_shapes()
def bilinear(x: ShapedTensor["...B m k"], y: ShapedTensor["...B k n"]) -> ShapedTensor["...B m n"]:
    return x @ y
```

### Dtype Annotations

Add a dtype kind — and optionally a bit width — anywhere in the descriptor:

```python
@check_tensor_shapes()
def safe_mean(x: ShapedTensor["float n k"]) -> ShapedTensor["float n"]:
    return x.mean(axis=-1)
```

Supported dtype tokens:

| Token | Accepted dtypes |
|-------|----------------|
| `bool` | boolean |
| `int`, `int8`, `int16`, `int32`, `int64` | signed integer |
| `uint`, `uint8`, `uint16`, `uint32`, `uint64` | unsigned integer |
| `integral` | any integer (signed or unsigned) |
| `float`, `float16`, `float32`, `float64` | real floating-point |
| `complex`, `complex64`, `complex128` | complex floating-point |
| `numeric` | any non-boolean numeric dtype |

These tokens are reserved and cannot be used as variable names.

### Optional and Nested Annotations

Annotations can be arbitrarily nested in tuples or lists. Mark an optional tensor with `| None`:

```python
@check_tensor_shapes()
def process(
        x: tuple[ShapedTensor["n k"], ShapedTensor["n"] | None],
        y: ShapedTensor["n 3"],
) -> ShapedTensor["n"]:
    a, b = x
    result = y.sum(axis=1)
    if b is not None:
        result = result + b
    return result
```

`NamedTuple` classes are also supported — apply the decorator to the class itself.

## API Reference

### `check_tensor_shapes`

```python
@check_tensor_shapes(
    constraints=None,
    ints_to_variables=True,
    check_mode=None,
    include_outer_variables=None,
    disable_union_warning=False,
)
```

Decorator that enables shape checking for a function, class `__init__`, or `NamedTuple` class.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constraints` | `list[str \| Callable]` | `None` | Extra constraints on shape variables. String expressions are evaluated (e.g. `"a == 2 * b"`); callables receive the variable dict and must return `bool`. Checked before *and* after the wrapped call. |
| `ints_to_variables` | `bool` | `True` | Promote `int` parameters to shape variables. |
| `check_mode` | `"always" \| "once" \| "never" \| None` | `None` | Per-function check mode, overrides the global setting. |
| `include_outer_variables` | `bool \| None` | `None` | Inherit shape variables from an enclosing `check_tensor_shapes` scope. Defaults to `False` for functions and `True` for `NamedTuple` instances. |
| `disable_union_warning` | `bool` | `False` | Suppress the warning about partially unsupported union types. |

---

### `set_global_check_mode`

```python
set_global_check_mode(mode: Literal["always", "once", "never"])
```

Set the global check mode for all `@check_tensor_shapes`-decorated functions. Per-function `check_mode` takes precedence when specified.

| Mode | Behaviour |
|------|-----------|
| `"always"` | Check every call (default) |
| `"once"` | Check each decorated function only on its first call |
| `"never"` | Disable all shape checks globally |

---

### `get_shape_variables`

```python
get_shape_variables(names: str) -> tuple[int | tuple[int, ...] | None, ...]
```

Return the current inferred values of shape variables. Must be called from inside a `@check_tensor_shapes`-wrapped function.

```python
@check_tensor_shapes()
def my_func(x: ShapedTensor["n k 3"]):
    n, k = get_shape_variables("n k")
    print(f"n={n}, k={k}")

my_func(np.zeros((10, 9, 3)))  # prints "n=10, k=9"
```

---

### `assert_shape_here`

```python
assert_shape_here(obj_or_shape, descriptor: str) -> None
```

Validate a tensor or shape tuple against a descriptor from inside a `@check_tensor_shapes`-wrapped function. Any new variables in the descriptor are registered for subsequent checks, including the function's return annotation.

```python
@check_tensor_shapes()
def my_func(x: ShapedTensor["n k"]) -> ShapedTensor["n m"]:
    y = some_operation(x)
    assert_shape_here(y, "n m")  # registers m; return annotation reuses it
    return y
```

---

### `label_tensor`

```python
label_tensor(tensor, label: str | Iterable[str], overwrite: bool = False) -> tensor
```

Attach one or more labels to a tensor. Labels registered with `register_label` can appear in shape descriptors and are matched against the tensor's labels at call time.

```python
from tensor_shape_assert import register_label, label_tensor

register_label("encoder_output")

z = label_tensor(encoder(x), "encoder_output")

@check_tensor_shapes()
def decode(z: ShapedTensor["encoder_output n d"]) -> ShapedTensor["n vocab"]:
    ...
```

---

### `register_label`

```python
register_label(label: str, constraint_fn: Callable[[array], bool] | None = None)
```

Register a custom label token for use in shape descriptors.

- If `constraint_fn` is `None`, the label is **unconstrained**: tensors must be explicitly tagged with `label_tensor` before being passed to a checked function.
- If `constraint_fn` is provided, the label behaves like a **dtype annotation**: any tensor whose descriptor contains this label is automatically checked by calling `constraint_fn(tensor)`. Constrained labels cannot be assigned via `label_tensor`.

---

### Trace Utilities

Use the trace API to inspect how shape variables are inferred — useful for debugging:

```python
from tensor_shape_assert import start_trace_recording, stop_trace_recording, trace_records_to_string

start_trace_recording()
my_func(np.zeros((10, 9, 3)))
records = stop_trace_recording()
print(trace_records_to_string(records))
```

| Function | Description |
|----------|-------------|
| `start_trace_recording()` | Begin capturing per-parameter variable assignments |
| `stop_trace_recording()` | Stop capturing and return the list of `TraceRecord` objects |
| `trace_records_to_string(records)` | Format records as an indented, human-readable string |

---

### Type-Safe Literal Syntax

For full static type-checker (MyPy, Pyright) compatibility, use `ShapedLiteral` and the pre-built framework aliases:

```python
import torch
from typing import Literal as L
from tensor_shape_assert import check_tensor_shapes, ShapedTorchLiteral, ShapedLiteral

@check_tensor_shapes()
def my_func(
        x: ShapedTorchLiteral[L["n k"]],
        y: ShapedTorchLiteral[L["k m"]],
) -> ShapedLiteral[torch.Tensor, L["n m"]]:
    return x @ y
```

| Alias | Array type |
|-------|------------|
| `ShapedTorchLiteral[L["..."]]` | `torch.Tensor` |
| `ShapedNumpyLiteral[L["..."]]` | `numpy.ndarray` |
| `ShapedLiteral[T, L["..."]]` | Any type `T` |

## Extended Example

Tuple inputs, optional parameters, and batch dimensions together:

```python
import torch
from tensor_shape_assert import check_tensor_shapes, ShapedTensor

@check_tensor_shapes()
def attention(
        query: ShapedTensor["...B heads seq_q d"],
        key_value: tuple[
            ShapedTensor["...B heads seq_kv d"],
            ShapedTensor["...B heads seq_kv d"],
        ],
        mask: ShapedTensor["...B 1 seq_q seq_kv"] | None = None,
) -> ShapedTensor["...B heads seq_q d"]:
    keys, values = key_value
    scores = query @ keys.transpose(-2, -1)  # (...B, heads, seq_q, seq_kv)
    if mask is not None:
        scores = scores + mask
    weights = scores.softmax(dim=-1)
    return weights @ values

# All of the following pass:
attention(
    query=torch.zeros(2, 8, 16, 64),
    key_value=(torch.zeros(2, 8, 32, 64), torch.zeros(2, 8, 32, 64)),
)

attention(
    query=torch.zeros(4, 2, 8, 16, 64),  # extra batch dim
    key_value=(torch.zeros(4, 2, 8, 32, 64), torch.zeros(4, 2, 8, 32, 64)),
    mask=torch.zeros(4, 2, 1, 16, 32),
)
```

## Compatibility

`tensor-shape-assert` works with any array library whose objects expose a `.shape` property:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [JAX](https://jax.readthedocs.io/)
- [TensorFlow](https://tensorflow.org/)
- Any library conforming to the [Python Array API standard](https://data-apis.org/array-api/latest/)

## License

This project is released under the Unlicense. See [LICENSE](LICENSE) for details.
```

Another benefit from using the typed version is that tooltips in VS Code are more helpful, as they can pass trough the ``Literal`` string. This way you can check the annotated shape without having to open the file with the annotated code.

## Known bugs
* [ ] ``get_shape_variables`` does not work if checks are disabled. This should be possible but give a performance warning, recommending not to use this feature in performance-critical applications.
* [x] Variable stack collects int values as variables and prints them, i.e. it produces error messages like ``"Shape torch.Size([1, 4, 4]) does not match descriptor (b, 4, 4) at position 0 based on already inferred variables {'b': 4, 'n': 304, 2: 2, 1: 1, 4: 4}"``

