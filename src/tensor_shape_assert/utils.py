class TensorShapeAssertError(RuntimeError):
    pass

def optional_to_int(s: str):
    try:
        return int(s)
    except ValueError:
        return s