from .utils import TensorShapeAssertError, optional_to_int

# define errors

class DescriptorValidationError(TensorShapeAssertError):
    pass

class VariableMatchingError(TensorShapeAssertError):
    pass

# util functions

def is_multi_dim_descriptor(s: str):
    return isinstance(s, str) and s.startswith('...')

# ignore all punctuation except its a required char

WHITESPACE_CHARS = "!\"#$%&',:;<=>?@[\\]^`{|}~"

def clean_up_descriptor(shape_descriptor: str):
    # remove ignored chars
    clean_shape_descriptor = shape_descriptor
    for ignored_char in WHITESPACE_CHARS:
        clean_shape_descriptor = clean_shape_descriptor.replace(ignored_char, ' ')

    # throw an error if there are more than 3 dots in a row
    if '....' in clean_shape_descriptor:
        raise DescriptorValidationError(
            f"Found a descriptor item that contains more than three dots "
            f"while parsing '{shape_descriptor}'."
        )
    
    # split into tokens temporarily
    tokens = clean_shape_descriptor.split(' ')

    # remove dots that don't form an elipsis
    tokens = [t.replace('.', '') if not t.startswith('...') else t for t in tokens]

    # remove multiple spaces
    tokens = [t for t in tokens if len(t) > 0]

    return ' '.join(tokens)


def split_to_descriptor_items(shape_descriptor: str):
    # split into individual strings and ints
    shape_descriptor = clean_up_descriptor(shape_descriptor)
    descriptor_items = shape_descriptor.split(" ")
    descriptor_items = tuple(optional_to_int(i) for i in descriptor_items)
    
    # there should be at most one multi dim descriptor
    mdd_idxs = [i for i, s in enumerate(descriptor_items) if is_multi_dim_descriptor(s)]
    
    if len(mdd_idxs) > 1:
        raise DescriptorValidationError(
            f"Found more than one descriptor item for multiple dimensions "
            f"while parsing '{shape_descriptor}'. Only a single multi-dim "
            f"descriptor is supported."
        )

    # split if there is a multi dim descriptor
    if len(mdd_idxs) == 1:
        mdd_idx = mdd_idxs[0]
        mdd = descriptor_items[mdd_idx]
        front_items = descriptor_items[:mdd_idx]
        back_items = descriptor_items[mdd_idx + 1:]
    else:
        mdd = None
        front_items = descriptor_items
        back_items = tuple()

    return front_items, back_items, mdd
    

def get_expected_shape_item(descriptor_item, shape_item, prev_var):
    if isinstance(descriptor_item, int):
        # if it is an integer, check shape is exactly that value
        return descriptor_item, descriptor_item
    
    elif descriptor_item == '*':
        # if its a wildcard, allow any value
        return None, None
    
    elif prev_var is None:
        # if no previous var is given, just return shape value
        return None, shape_item
    
    else:
        # if variable is already set, values must matach
        return prev_var, prev_var

def descriptor_items_to_string(descriptor_items: tuple[str | int]) -> str:
    return str(tuple(descriptor_items)).replace("'", "")

def descriptor_to_variables(shape_descriptor, shape, variables=None):
    front_items, back_items, multi_dim_item = split_to_descriptor_items(shape_descriptor)

    if variables is None:
        variables = dict()

    # build check list
    if multi_dim_item is not None:
        

        if len(front_items) + len(back_items) < len(shape):
            shape_multi = [shape[len(front_items):len(shape) - len(back_items)]]
            descriptor_items = (*front_items, multi_dim_item, *back_items)
        else:
            shape_multi = []
            descriptor_items = (*front_items, *back_items)

        shape_items = (
            *shape[:len(front_items)],
            *shape_multi,
            *shape[len(shape) - len(back_items):]
        )
    else:
        descriptor_items = front_items
        shape_items = shape

    # check if number of dimensions is correct
    if len(descriptor_items) != len(shape_items):
        raise VariableMatchingError(
            f"Shape {shape} has the wrong number of matcher dimensions "
            f"(dims={len(shape_items)}) to be matched with descriptor "
            f"{descriptor_items_to_string(descriptor_items)} "
            f"(dims={len(descriptor_items)})."
        )

    # check and infer
    desc_shape_pairs = zip(descriptor_items, shape_items)
    for i, (desc_item, shape_item) in enumerate(desc_shape_pairs):
        expected_value, resulting_value = get_expected_shape_item(
            descriptor_item=desc_item,
            shape_item=shape_item,
            prev_var=variables.get(desc_item)
        )

        if expected_value is not None and shape_item != expected_value:
            raise VariableMatchingError(
                f"Shape {shape} does not match descriptor "
                f"{descriptor_items_to_string(descriptor_items)} "
                f"at position {i} based on already inferred variables "
                f"{variables}."
            )
            
        if resulting_value is not None:
            variables[desc_item] = resulting_value

    return variables
