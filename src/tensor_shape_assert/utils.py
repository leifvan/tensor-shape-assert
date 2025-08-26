from array_api_compat import array_namespace

class TensorShapeAssertError(RuntimeError):
    pass

def optional_to_int(s: str):
    try:
        return int(s)
    except ValueError:
        return s


def check_if_dtype_matches(obj, kind, bits):
    api = array_namespace(obj)
    
    # dtype does not match
    if not api.isdtype(obj.dtype, kind):
        return False

    # bits not given -> accept
    if bits is None:
        return True

    # check bits via info function
    if api.isdtype(obj.dtype, "integral"):
        return api.iinfo(obj.dtype).bits == bits
    elif api.isdtype(obj.dtype, "real floating"):
        return api.finfo(obj.dtype).bits == bits
    elif api.isdtype(obj.dtype, "complex floating"):
        return 2 * api.finfo(obj.dtype).bits == bits
    else:
        raise TensorShapeAssertError(
            f"Dtype '{obj.dtype}' does not support bit checking."
        )
