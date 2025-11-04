
# define errors

class TensorShapeAssertError(RuntimeError): pass

class MalformedDescriptorError(TensorShapeAssertError): pass
class NoVariableContextExistsError(TensorShapeAssertError): pass
class AnnotationMatchingError(TensorShapeAssertError): pass
class VariableConstraintError(TensorShapeAssertError): pass
class DtypeConstraintError(TensorShapeAssertError): pass
class UnionTypeUnsupportedError(TensorShapeAssertError): pass
class DescriptorValidationError(TensorShapeAssertError): pass
class VariableMatchingError(TensorShapeAssertError): pass

# define warnings

class CheckDisabledWarning(RuntimeWarning): pass
