from types import ModuleType
from importlib import import_module

NAME_LIBRARY_MAP = {
    "torch": "array_api_compat.torch",
    "numpy": "array_api_compat.numpy",
    "cupy": "array_api_compat.cupy",
    "dask": "array_api_compat.dask.array",
    "jax": "jax.numpy",
    "ndonnx": "ndonnx",
    "sparse": "sparse",
}

def get_library_by_name(name: str) -> ModuleType:
    try:
        module_name = NAME_LIBRARY_MAP[name]
        return import_module(module_name)
    except KeyError:
        raise ValueError(f"Unsupported library: {name}")