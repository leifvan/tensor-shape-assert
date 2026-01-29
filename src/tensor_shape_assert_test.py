# pyright: reportArgumentType=false
# pyright: reportReturnType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportPossiblyUnboundVariable=false
# mypy: ignore-errors

import sys
import unittest
import warnings
import os
from multiprocessing import Queue
from typing import Callable, NamedTuple, TYPE_CHECKING

# import these from public module
from tensor_shape_assert import (
    check_tensor_shapes,
    get_shape_variables,
    assert_shape_here,
    set_global_check_mode,
    ShapedTensor,
    ScalarTensor,
    start_trace_recording,
    stop_trace_recording,
    trace_records_to_string
)

from tensor_shape_assert.errors import (
    MalformedDescriptorError,
    UnionTypeUnsupportedError,
    CheckDisabledWarning,
)

from tensor_shape_assert.utils import TensorShapeAssertError
from tensor_shape_assert.wrapper import (
    NoVariableContextExistsError,
    VariableConstraintError
)
from test_utils import get_library_by_name

# read library to be used from env
xp = get_library_by_name(os.environ["TSA_TEST_LIBRARY"])


def library_has_types(type_names: list[str]) -> bool:
    return all(hasattr(xp, t) for t in type_names)

class Test1DAnnotationsKeyword(unittest.TestCase):
    def test_constant_1d_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5"]) -> ShapedTensor["2"]:
            return x[:2]
        
        x = xp.zeros(5)
        self.assertTupleEqual(test(x=x).shape, (2,))

    def test_constant_1d_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5"]) -> ShapedTensor["2"]:
            return x[:2]
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(6)
            test(x=x) 
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((1, 5))
            test(x=x)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((5, 1))
            test(x=x)
            
    def test_variable_1d_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["b"]:
            return x + 1
        
        x = xp.zeros(8)
        self.assertTupleEqual(test(x=x).shape, x.shape)
        self.assertTrue(xp.all(x != test(x=x)))
        
    def test_variable_1d_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["b"]:
            return x[:2]
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            test(x=x)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((10, 2))
            test(x=x)

    def test_no_output_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]):
            return x + 1
        
        x = xp.zeros(8)
        self.assertTupleEqual(test(x=x).shape, x.shape)
        self.assertTrue(xp.all(x != test(x=x)))

    def test_no_output_annotation_with_tuple(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]):
            return (x + 1, x + 2)
        
        x = xp.zeros(8)
        self.assertTupleEqual(test(x=x)[0].shape, x.shape)
        self.assertTrue(xp.all(x != test(x=x)[0]))

    def test_wrong_output_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["1"]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            self.assertTupleEqual(test(x=x).shape, x.shape)
            self.assertTrue(xp.all(x != test(x=x)))

    def test_wrong_output_annotation_one_item(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1"], ShapedTensor["2"]]:
            return x + 1 # type: ignore
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            test(x=x)

    def test_wrong_output_annotation_multiple_items(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1"], ShapedTensor["2"], ShapedTensor["3"]]:
            return (x + 1, x + 2) # type: ignore
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            test(x=x)

    def test_wrong_output_annotation_more_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["1 b 2 3"]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            self.assertTupleEqual(test(x=x).shape, x.shape)
            self.assertTrue(xp.all(x != test(x=x)))

    def test_wrong_output_annotation_less_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 1 1 2"]) -> ShapedTensor["1"]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((8, 1, 1, 2))
            self.assertTupleEqual(test(x=x).shape, x.shape)
            self.assertTrue(xp.all(x != test(x=x)))

    def test_wrong_output_annotation_tuple(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1"], ShapedTensor["b"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            self.assertTupleEqual(test(x=x)[0].shape, x.shape)
            self.assertTrue(xp.all(x != test(x=x)[0]))

    def test_wrong_output_annotation_tuple_more_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1 b"], ShapedTensor["2 b"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(8)
            self.assertTupleEqual(test(x=x)[0].shape, x.shape)
            self.assertTrue(xp.all(x != test(x=x)[0]))

    def test_wrong_output_annotation_tuple_less_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 1 2 3"]) -> tuple[ShapedTensor["b"], ShapedTensor["b"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((8, 1, 2, 3))
            self.assertTupleEqual(test(x=x)[0].shape, x.shape)
            self.assertTrue(xp.all(x != test(x=x)[0]))
            
class TestNDAnnotationsKeyword(unittest.TestCase):
    def test_constant_nd_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5 1"], y: ShapedTensor["1 3"]) -> ShapedTensor["2"]:
            return x[:2, 0] + y[0, 1:]
        
        x = xp.zeros((5, 1))
        y = xp.zeros((1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (2,))
        
    def test_constant_nd_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5 4 3"], y: ShapedTensor["3 4 5"]) -> ShapedTensor["3"]:
            return x[1, 2, :] + y[:, 1, 2]
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((5, 4, 2))
            y = xp.zeros((3, 4, 5))
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((5, 4, 3))
            y = xp.zeros((3, 4))
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros(5)
            y = xp.zeros((3, 4, 5))
            test(x=x, y=y)
            
    def test_variable_nd_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 3 2"], y: ShapedTensor["3 c 2"]) -> ShapedTensor["3 c"]:
            return xp.sum(x[:1, :1, :] + y, axis=2)
        
        x = xp.zeros((17, 3, 2))
        y = xp.zeros((3, 19, 2))
        self.assertTupleEqual(test(x=x, y=y).shape, (3, 19))
        
        x = xp.zeros((1, 3, 2))
        y = xp.zeros((3, 19, 2))
        self.assertTupleEqual(test(x=x, y=y).shape, (3, 19))
        
        x = xp.zeros((17, 3, 2))
        y = xp.zeros((3, 12, 2))
        self.assertTupleEqual(test(x=x, y=y).shape, (3, 12))
        
    def test_variable_nd_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 3 2"], y: ShapedTensor["3 c 2"]) -> ShapedTensor["3 c"]:
            return (x[:1, :1] + y).sum(axis=2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((17, 2, 2))
            y = xp.zeros((3, 19, 2))
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((17, 3))
            y = xp.zeros((3, 19, 2))
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((17, 3, 2))
            y = xp.zeros((3, 2))
            test(x=x, y=y)
            
            
class TestArbitraryBatchDimAnnotationsKeyword(unittest.TestCase):
    def test_constant_nd_arbitrary_batch_input_shape_checked_front(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["... 5 1"], y: ShapedTensor["... 1 3"]) -> ShapedTensor["... 2"]:
            return x[..., :2, 0] + y[..., 0, 1:]
        
        x = xp.zeros((5, 1))
        y = xp.zeros((1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (2,))
        
        x = xp.zeros((1, 5, 1))
        y = xp.zeros((1, 1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (1, 2))
        
        x = xp.zeros((32, 5, 1))
        y = xp.zeros((32, 1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (32, 2))
        
        x = xp.zeros((9, 8, 7, 5, 1))
        y = xp.zeros((9, 8, 7, 1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (9, 8, 7, 2))

    def test_constant_nd_arbitrary_batch_input_shape_checked_middle(self):
        @check_tensor_shapes()
        def test(
                x: ShapedTensor["5 ... 1"],
                y: ShapedTensor["1 ... 3"]
        ) -> ShapedTensor["... 2"]:
            z = x[4, ..., 0, None]
            return xp.concat([z, z], axis=-1) + y[0, ..., 1:]
        
        x = xp.zeros((5, 1))
        y = xp.zeros((1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (2,))
        
        x = xp.zeros((5, 1, 1))
        y = xp.zeros((1, 1, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (1, 2))
        
        x = xp.zeros((5, 32, 1))
        y = xp.zeros((1, 32, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (32, 2))
        
        x = xp.zeros((5, 9, 8, 7, 1))
        y = xp.zeros((1, 9, 8, 7, 3))
        self.assertTupleEqual(test(x=x, y=y).shape, (9, 8, 7, 2))
        
    def test_constant_nd_arbitrary_batch_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["... 5 1"], y: ShapedTensor["... 1 3 2"]) -> ShapedTensor["... 2"]:
            return xp.zeros((6, 5, 4, 3, 1))
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((1, 5, 2))
            y = xp.zeros((1, 1, 3))
            test(x=x, y=y)
        
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((15, 5, 1))
            y = xp.zeros((20, 1, 3))
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = xp.zeros((15, 5, 1))
            y = xp.zeros((20, 1, 3, 2))
            test(x=x, y=y)


class TestNamedBatchDimensions(unittest.TestCase):
    def test_correct_1_batch_dimension_equal_length(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3 2"]
        ) -> ShapedTensor["...B 2 2"]:
            return x @ y
        
        test(
            x=xp.zeros((12, 2, 3)),
            y=xp.zeros((12, 3, 2))
        )

    def test_correct_1_batch_dimension_x_longer_y(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3"]
        ) -> ShapedTensor["...B 2 3"]:
            return x * y[..., None, :]
        
        test(
            x=xp.zeros((12, 2, 3)),
            y=xp.zeros((12, 3))
        )

    def test_correct_1_batch_dimension_y_longer_x(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2"],
            y: ShapedTensor["...B 2 3"]
        ) -> ShapedTensor["...B 2 3"]:
            return x[..., None] * y
        
        test(
            x=xp.zeros((12, 2)),
            y=xp.zeros((12, 2, 3))
        )

    def test_correct_2_batch_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
            return x @ y
        test(
            x=xp.zeros((12, 3, 2, 3)),
            y=xp.zeros((12, 3, 3, 2))
        )

    def test_correct_2_batch_dimensions_tuple_output(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3 2"]
        ) -> tuple[
            ShapedTensor["...B 2 2"],
            ShapedTensor["...B"]
        ]:
            return x @ y, (x @ y).sum(axis=(-1, -2))
        
        test(
            x=xp.zeros((12, 3, 2, 3)),
            y=xp.zeros((12, 3, 3, 2))
        )

    def test_wrong_length_batch_dimensions_input(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return x @ y
            test(
                x=xp.zeros((12, 2, 3)),
                y=xp.zeros((12, 3, 2, 3))
            )

    def test_wrong_length_batch_dimensions_output(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return (x @ y)[0]
            test(
                x=xp.zeros((12, 3, 2, 3)),
                y=xp.zeros((12, 3, 2, 3))
            )

    def test_wrong_size_batch_dimensions_input(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return x @ y
            test(
                x=xp.zeros((12, 4, 2, 3)),
                y=xp.zeros((12, 3, 2, 3))
            )

    def test_wrong_size_batch_dimensions_output(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return (x @ y)[:5]
            test(
                x=xp.zeros((12, 4, 2, 3)),
                y=xp.zeros((12, 4, 2, 3))
            )

class TestClassFunctionality(unittest.TestCase):
    def test_method_annotation(self):
        class Test:
            @check_tensor_shapes()
            def my_method(
                self,
                x: ShapedTensor["b 3"],
                y: ShapedTensor["b"]
            ):
                return x + y[:, None]
        
        Test().my_method(
            x=xp.zeros((17, 3)),
            y=xp.zeros(17)
        )
        
        with self.assertRaises(TensorShapeAssertError):
            Test().my_method(
                x=xp.zeros((17, 4)),
                y=xp.zeros(17)
            )

    def test_constructor_annotation(self):
        class Test:
            @check_tensor_shapes()
            def __init__(self, x: ShapedTensor["b 3"], y: ShapedTensor["b"]):
                self.z = x + y[:, None]
        
        Test(
            x=xp.zeros((17, 3)),
            y=xp.zeros(17)
        )

        with self.assertRaises(TensorShapeAssertError):
            Test(
                x=xp.zeros((17, 4)),
                y=xp.zeros(17)
            )

    def test_inherited_method_annotation(self):
        class Test:
            @check_tensor_shapes()
            def my_method(
                self,
                x: ShapedTensor["b 3"],
                y: ShapedTensor["b"]
            ) -> ShapedTensor["b 3"]:
                return x + y[:, None]
            
        class SubTest(Test):
            pass

        SubTest().my_method(
            x=xp.zeros((17, 3)),
            y=xp.zeros(17)
        )

    def test_overridden_method_annotation(self):
        class Test:
            @check_tensor_shapes()
            def my_method(
                self,
                x: ShapedTensor["b 3"],
                y: ShapedTensor["b"]
            ) -> ShapedTensor["b 3"]:
                return x + y[:, None]
            
        class SubTest(Test):
            @check_tensor_shapes()
            def my_method(
                self,
                x: ShapedTensor["b 3"],
                y: ShapedTensor["b"]
            ) -> ShapedTensor["b"]:
                return super().my_method(x=x, y=y).sum(axis=1) 
        
        SubTest().my_method(
            x=xp.zeros((17, 3)),
            y=xp.zeros(17)
        )


class TestTensorDescriptor(unittest.TestCase):
    @check_tensor_shapes()
    def test_multiple_whitespaces_ignored(self):
        def test(x: ShapedTensor["a   b c  "]) -> ShapedTensor[" b   "]:
            return xp.mean(x, axis=0)[:, 0]
        test(x=xp.zeros((3, 4, 5)))
    
    def test_commas_ignored(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a,b,c"]) -> ShapedTensor["b,,"]:
            return xp.mean(x, axis=0)[:, 0]
        test(x=xp.zeros((3, 4, 5)))

    def test_parentheses_ignored(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b[c"]) -> ShapedTensor["b]"]:
            return xp.mean(x, axis=0)[:, 0]
        test(x=xp.zeros((3, 4, 5)))

        
class TestMisc(unittest.TestCase):
    def test_instantiating(self):
        with self.assertRaises(RuntimeError):
            ShapedTensor() # type: ignore

    def test_misc_annotations_ignored(self):
        @check_tensor_shapes()
        def test1(x: ShapedTensor["1"]) -> str:
            return "hi"
        
        test1(x=xp.zeros(1))

        @check_tensor_shapes()
        def test2(x: ShapedTensor["1"]) -> tuple[int, int, str]:
            return 1, 2, "hi"

        test2(x=xp.zeros(1))

        @check_tensor_shapes()
        def test3(x: ShapedTensor["1"]) -> list[str]:
            return ["hi", "bye"]

        test3(x=xp.zeros(1))

        @check_tensor_shapes()
        def test4(x: ShapedTensor["1"]) -> Callable[[str, str], int]:
            def _test(a: str, b: str) -> int:
                return len(a) + len(b)
            return _test
        
        test4(x=xp.zeros(1))

    # def test_helpful_error_message_when_brackets_missing(self):
        
    #     with self.assertRaises(TensorShapeAssertError) as cm:
    #         @check_tensor_shapes
    #         def test(x: ShapedTensor["a"]) -> ShapedTensor["b"]:
    #             return x[1:]
            
    #         test(x=xp.zeros(5))
        
    #     self.assertIn("Maybe you forgot brackets", str(cm.exception))

    def test_brackets_missing_uses_defaults(self):
    
        @check_tensor_shapes
        def test(x: ShapedTensor["a"]) -> ShapedTensor["b"]:
            return x[1:]

        test(x=xp.zeros(5)) # type: ignore
        self.assertTrue(True)
            

    def test_union_type_error(self):
        with self.assertRaises(UnionTypeUnsupportedError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["a"] | str):
                return x
            
            test(x=xp.zeros(5))

class TestGetVariableValuesFromCurrentContext(unittest.TestCase):
    def test_single_context(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b"]) -> ShapedTensor["a"]:
            a, b = get_shape_variables("a b")
            self.assertTupleEqual(x.shape, (a, b))
            return x.sum(axis=1)

        test(x=xp.zeros((5, 6)))

    def test_nested_context(self):
        x_input = xp.zeros((5, 6, 7))

        @check_tensor_shapes()
        def sum_plus_one(x: ShapedTensor["a b"]) -> ShapedTensor["b"]:
            a, b = get_shape_variables("a b")
            self.assertTupleEqual(x_input.shape[1:], (a, b))
            return x.sum(axis=0) + 1

        @check_tensor_shapes()
        def test(x: ShapedTensor["a b c"]) -> ShapedTensor["c"]:
            a, b, c = get_shape_variables("a b c")
            self.assertTupleEqual(x_input.shape, (a, b, c))
            y = sum_plus_one(x=x[0, :, :])
            return y
        
        test(x=x_input)

    def test_error_on_no_context(self):
        with self.assertRaises(NoVariableContextExistsError):
            a, b = get_shape_variables("a b")
            print(a, b)

    def test_unknown_variable_is_none(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b"]) -> ShapedTensor["a"]:
            c = get_shape_variables("c")[0]
            self.assertIsNone(c)
            return x.sum(axis=1)
        
        test(x=xp.zeros((5, 6)))

    def test_extra_characters(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b"]) -> ShapedTensor["a"]:
            a, b = get_shape_variables(" a  ] [[b    ")
            self.assertTupleEqual(x.shape, (a, b))
            return x.sum(axis=1)
        
        test(x=xp.zeros((5, 6)))

    def test_state_does_not_collect_ints(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 2"]):
            k = get_shape_variables("2")[0]
            self.assertIsNone(k)
            return x

        test(xp.zeros((3, 2)))

    def test_state_has_batch_dimension(self):
        @check_tensor_shapes()
        def test1(x: ShapedTensor["... 4"]):
            batch = get_shape_variables("...")[0]
            self.assertTupleEqual(batch, (1, 2, 3))
            return x

        test1(xp.zeros((1, 2, 3, 4)))

        @check_tensor_shapes()
        def test2(x: ShapedTensor["4 ..."]):
            batch = get_shape_variables("...")[0]
            self.assertTupleEqual(batch, (3, 2, 1))
            return x

        test2(xp.zeros((4, 3, 2, 1)))

        @check_tensor_shapes()
        def test3(x: ShapedTensor["1 ... 4"]):
            batch = get_shape_variables("...")[0]
            self.assertTupleEqual(batch, (2, 3))
            return x

        test3(xp.zeros((1, 2, 3, 4)))


class TestPositionalAndMixedArguments(unittest.TestCase):
    def test_positional_only(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["c 3"], y: ShapedTensor["3 c"]) -> ShapedTensor["c c"]:
            return x @ y
        test(xp.zeros((5, 3)), xp.zeros((3, 5)))

    def test_mixed_positional_and_kwargs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["c 3"], y: ShapedTensor["3 c"]) -> ShapedTensor["c c"]:
            return x @ y
        test(xp.zeros((5, 3)), y=xp.zeros((3, 5)))


class TestArbitrarilyNestedInputs(unittest.TestCase):
    def test_flat_input_tuples(self):
        @check_tensor_shapes()
        def test(x: tuple[
                ShapedTensor["a 2"],
                ShapedTensor["b 2"],
        ], *args) -> ShapedTensor["a"]:
            return (x[0] * x[1][0, :][0]).sum(axis=1)
        
        test((
            xp.zeros((10, 2)),
            xp.zeros((9, 2))
        ))

        test((
            xp.zeros((28, 2)),
            xp.zeros((49, 2))
        ))

        test([
            xp.zeros((10, 2)),
            xp.zeros((9, 2))
        ])

        with self.assertRaises(TensorShapeAssertError):
            test((
                xp.zeros((10, 3)),
                xp.zeros((9, 2))
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                xp.zeros((10, 2)),
                xp.zeros(2)
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                xp.zeros((10, 2)),
                xp.zeros((3, 2)),
                xp.zeros((2, 2))
            ))

        with self.assertRaises(TensorShapeAssertError):
            test(
                (xp.zeros((10, 2)),)
            )

        with self.assertRaises(TensorShapeAssertError):
            test(
                xp.zeros((10, 2)),
                xp.zeros((9, 2))
            )

        with self.assertRaises(AttributeError):
            test((
                (xp.zeros((10, 2)),),
                (xp.zeros((10, 2)),),
            ))

        with self.assertRaises(TensorShapeAssertError):
            test(
                xp.zeros((10, 2)),
                None
            )
             
    def test_complex_input_tuples(self):

        if not library_has_types(["complex64", "complex128"]):
            self.skipTest(f"'{xp.__name__}' doesn't support complex types.")

        @check_tensor_shapes()
        def test(
                x: tuple[
                    tuple[
                        tuple[ShapedTensor["a"], ShapedTensor["b"]],
                        ShapedTensor["a 2"]
                    ],
                    ShapedTensor["b 2"],
                ]
        ) -> ShapedTensor["a"]:
            return x[0][0][0]
        
        test((
            (
                (xp.zeros(4), xp.zeros(3)),
                xp.zeros((4, 2))
            ),
            xp.zeros((3, 2))
        ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                (
                    (xp.zeros(4), xp.zeros(4)),
                    xp.zeros((4, 2))
                ),
                xp.zeros((3, 2))
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                (
                    (xp.zeros(4), xp.zeros(3)),
                    xp.zeros((4, 3))
                ),
                xp.zeros((3, 2))
            ))
        
class TestLocalShapeChecking(unittest.TestCase):
    def test_local_shape_check_positive(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"], y: ShapedTensor["...B 3"]) -> ShapedTensor["...B"]:
            z = y[..., 0] + x[0, :]
            assert_shape_here(z, "...B")
            return z

        test(x=xp.zeros((2, 1)), y=xp.zeros((5, 4, 3)))

    def test_local_shape_check_negative(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"], y: ShapedTensor["...B 3"]) -> ShapedTensor["...B"]:
            z = y[..., 0] + x[0, :]
            assert_shape_here(z, "a")
            return z

        with self.assertRaises(TensorShapeAssertError):
            test(x=xp.zeros((2, 1)), y=xp.zeros((5, 4, 3)))

    def test_local_shape_check_updates_context(self):

        @check_tensor_shapes(ints_to_variables=False)
        def test(
                x: ShapedTensor["a"],
                y: ShapedTensor["b"],
                c: int | None
        ) -> ShapedTensor["c"]:
            if c is not None:
                assert_shape_here((c,), "c")

            f = xp.zeros((x.shape[0] * y.shape[0], 5))
            return xp.sum(f, axis=1)
        
        test(xp.zeros(6), xp.zeros(7), c=42)
        test(xp.zeros(6), xp.zeros(7), c=None)

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros(6), xp.zeros(7), c=41)

        
class TestOptionalShapeAnnotation(unittest.TestCase):
    def test_optional_input_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"] | None) -> ShapedTensor["1"]:
            if x is not None:
                return x[0, :]
            else:
                return xp.zeros(1)
        
        test(xp.zeros((2, 1)))
        test(None)

    def test_warn_optional_output_tuple(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"] | None) -> tuple[ShapedTensor["1"]] | None:
            if x is not None:
                return x[0, :]
            else:
                return (xp.zeros(1), )
        
        with self.assertWarns(RuntimeWarning):
            test(xp.zeros((2, 1)))        

class TestVariableConstraints(unittest.TestCase):
    def test_lambda_constraints(self):
        @check_tensor_shapes(constraints=[lambda v: v['a'] + v['b']==5])
        def test(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> float:
            return a[0] + b[0]
        
        test(xp.zeros(3), xp.zeros(2))
        test(xp.zeros(1), xp.zeros(4))

        with self.assertRaises(VariableConstraintError):
            test(xp.zeros(3), xp.zeros(3))

    def test_str_expression_constraints(self):
        @check_tensor_shapes(constraints=["a + b = c"])
        def test1(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["c"]:
            return xp.concat([a, b], axis=0)
        
        test1(xp.zeros(3), xp.zeros(2))
        test1(xp.zeros(1), xp.zeros(4))

        @check_tensor_shapes(constraints=["a + b = c"])
        def test2(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["c"]:
            return xp.concat([a, b, xp.zeros(1)], axis=0)
        
        with self.assertRaises(VariableConstraintError):
            test2(xp.zeros(3), xp.zeros(3))

    def test_autogen_constraints(self):
        @check_tensor_shapes(experimental_enable_autogen_constraints=True)
        def test1(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["a+b"]:
            return xp.concat([a, b], axis=0)

        test1(xp.zeros(3), xp.zeros(2))
        test1(xp.zeros(1), xp.zeros(4))

        @check_tensor_shapes(experimental_enable_autogen_constraints=True)
        def test2(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["a+b"]:
            return xp.concat([a, b, xp.zeros(1)], axis=0)
        
        with self.assertRaises(VariableConstraintError):
            test2(xp.zeros(3), xp.zeros(3))

class TestIntToVariables(unittest.TestCase):
    def test_int_to_variable(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3"], m: int) -> ShapedTensor["m"]:
            return x.sum(axis=0)[:, 0]

        test(xp.zeros((5, 4, 3)), m=4)
        test(xp.zeros((5, 2, 3)), m=2)

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((5, 4, 3)), m=3)        

    def test_int_to_variable_deactivated(self):
        @check_tensor_shapes(ints_to_variables=False)
        def test(x: ShapedTensor["n m 3"], m: int) -> ShapedTensor["m"]:
            return x.sum(axis=0)[:, 0]

        test(xp.zeros((5, 4, 3)), m=4)
        test(xp.zeros((5, 2, 3)), m=2)
        test(xp.zeros((5, 4, 3)), m=3)

    def test_strict_int_comparison(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3"], m: bool) -> ShapedTensor["m"]:
            return x.sum(axis=0)[:, 0]
        
        test(xp.zeros((5, 4, 3)), m=True)

class TestDtypeAnnotationTorch(unittest.TestCase):
    def test_specified_int_types(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["int16 n m 3"]) -> ShapedTensor["int32 m"]:
            return xp.astype(xp.sum(xp.sum(x, axis=2), axis=0), xp.int32)

        test(xp.zeros((1, 2, 3), dtype=xp.int16))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.int32))

    def test_unspecificed_int_types(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["integral n m 3"]) -> ShapedTensor["int m"]:
            return xp.astype(xp.sum(xp.sum(x, axis=2), axis=0), xp.int32)
        
        test(xp.zeros((1, 2, 3), dtype=xp.int16))
        test(xp.zeros((1, 2, 3), dtype=xp.uint8))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.float32))

    def test_specified_float_types(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["float64 n m 3"]) -> ShapedTensor["float32 m"]:
            return xp.astype(xp.mean(x.sum(axis=2), axis=0), xp.float32)
        
        test(xp.zeros((1, 2, 3), dtype=xp.float64))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.float32))

    def test_unspecified_float_types(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["float n m 3"]) -> ShapedTensor["float m"]:
            return xp.astype(xp.mean(x.sum(axis=2), axis=0), xp.float32)
        
        test(xp.zeros((1, 2, 3), dtype=xp.float64))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.int32))

    def test_specified_complex_type(self):

        if not library_has_types(["complex64", "complex128"]):
            self.skipTest(f"'{xp.__name__}' doesn't support complex types.")

        @check_tensor_shapes()
        def test(x: ShapedTensor["complex128 n m 3"]) -> ShapedTensor["float32 m"]:
            return xp.astype(x.sum(axis=(0, 2)).real, xp.float32)

        test(xp.zeros((1, 2, 3), dtype=xp.complex128))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.float32))

    def test_unspecified_complex_type(self):

        if not library_has_types(["complex64", "complex128"]):
            self.skipTest(f"'{xp.__name__}' doesn't support complex types.")

        @check_tensor_shapes()
        def test(x: ShapedTensor["complex n m 3"]) -> ShapedTensor["float32 m"]:
            return xp.astype(x.sum(axis=(0, 2)).real, xp.float32)
        
        test(xp.zeros((1, 2, 3), dtype=xp.complex128))
        test(xp.zeros((1, 2, 3), dtype=xp.complex64))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.float64))

    def test_ignored_type(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3"], rt) -> ShapedTensor["float32 m"]:
            return xp.astype(x.sum(axis=(0, 2)), rt)
        
        test(xp.zeros((1, 2, 3), dtype=xp.float64), rt=xp.float32)

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.float32), rt=xp.float64)

    def test_bool_with_bitsize_raises_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["bool8 n m 3"], rt) -> ShapedTensor["float32 m"]:
            return xp.astype(x.sum(axis=(0, 2)), rt)

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.bool), rt=xp.float64)

    def test_dtype_is_position_agnostic(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["  n m int: 3 "]) -> ShapedTensor["m  |float32|"]:
            return xp.astype(x.sum(axis=(0, 2)), xp.float32)
        
        test(xp.zeros((1, 2, 3), dtype=xp.int64))
        test(xp.zeros((1, 2, 3), dtype=xp.int32))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((1, 2, 3), dtype=xp.float64))

    def test_fails_with_multiple_dtype_descriptors(self):
        with self.assertRaises(MalformedDescriptorError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["float n m complex 3"]) -> ShapedTensor["m"]:
                return xp.astype(xp.median(x.sum(axis=2), dim=0)[0], xp.int32)

    def test_scalar_dtype_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n"]) -> ShapedTensor["float32"]:
            return x.sum()
        
        test(xp.zeros(5, dtype=xp.float32))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros(7, dtype=xp.int32))


class TestNamedTupleSupport(unittest.TestCase):
    def test_named_tuple(self):
        
        @check_tensor_shapes()
        class MyTuple(NamedTuple):
            p: ShapedTensor["n m 3"]
            q: ShapedTensor["n 1 3"]
            num: int

        MyTuple(p=xp.zeros((5, 4, 3)), q=xp.zeros((5, 1, 3)), num=5)
        MyTuple(xp.zeros((10, 1, 3)), xp.zeros((10, 1, 3)), num=8)

        with self.assertRaises(TensorShapeAssertError):
            MyTuple(p=xp.zeros((5, 4, 3)), q=xp.zeros((5, 2, 3)), num=5)


class TestCheckMode(unittest.TestCase):
    def tearDown(self) -> None:
        # reset it here just to be safe
        set_global_check_mode('always')
        return super().tearDown()

    def test_always_checked_by_default(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))
        
        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 1)))

    def test_once_checked_global_ignores_errors(self):
        set_global_check_mode('once')

        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))
        test(xp.zeros((4, 3, 1)))
        test(xp.zeros((4, 3, 3)))

    def test_once_checked_global_detects_first_error(self):
        set_global_check_mode('once')

        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 1)))

        set_global_check_mode('always')

    def test_never_global_detects_no_errors(self):
        set_global_check_mode('never')

        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))
        test(xp.zeros((4, 3, 1)))
        test(xp.zeros((4, 3, 3)))

        set_global_check_mode('always')

    def test_never_global_ignores_assert_shape_here(self):
        set_global_check_mode('never')

        @check_tensor_shapes()
        def test(x) -> ShapedTensor["2"]:
            assert_shape_here(x, "m n 2")
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))
        test(xp.zeros((4, 3, 1)))
        test(xp.zeros((4, 3, 3)))

        set_global_check_mode('always')

    def test_local_always_overrides_global_never(self):
        set_global_check_mode('never')

        @check_tensor_shapes(check_mode='always')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        @check_tensor_shapes()
        def test2(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 1)))
        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 3)))

        test2(xp.zeros((4, 3, 1)))

        set_global_check_mode('always')

    def test_local_always_overrides_global_once(self):
        set_global_check_mode('once')

        @check_tensor_shapes(check_mode='always')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        @check_tensor_shapes()
        def test2(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 1)))
        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 3)))

        with self.assertRaises(TensorShapeAssertError):
            test2(xp.zeros((4, 3, 1)))

        test2(xp.zeros((4, 3, 5)))

        set_global_check_mode('always')

    def test_local_never_overrides_global_always(self):

        set_global_check_mode('always')

        @check_tensor_shapes(check_mode='never')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        @check_tensor_shapes()
        def test2(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))
        test(xp.zeros((4, 3, 1)))
        test(xp.zeros((4, 3, 3)))

        with self.assertRaises(TensorShapeAssertError):
            test2(xp.zeros((4, 3, 1)))
        with self.assertRaises(TensorShapeAssertError):
            test2(xp.zeros((4, 3, 3)))

    def test_local_once_overrides_global_always(self):

        set_global_check_mode('always')

        @check_tensor_shapes(check_mode='once')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))
        
        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 1)))

        test(xp.zeros((4, 3, 5)))
        test(xp.zeros((4, 3, 3)))

    def test_invalid_local_check_mode_raises(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes(check_mode='invalid_mode')
            def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
                return x.sum(axis=(0, 1))
        
    def test_invalid_global_check_mode_raises(self):
        with self.assertRaises(TensorShapeAssertError):
            set_global_check_mode("invalid_mode")

    def test_assert_shape_here_respects_global_check_mode(self):
        set_global_check_mode('never')

        @check_tensor_shapes()
        def test(x) -> ShapedTensor["2"]:
            assert_shape_here(x, "m n 2")
            return x.sum(axis=(0, 1))
        
        test(xp.zeros((4, 3, 2)))
        test(xp.zeros((4, 3, 1)))
        test(xp.zeros((4, 3, 3)))

        set_global_check_mode('always')

        @check_tensor_shapes()
        def test2(x) -> ShapedTensor["2"]:
            assert_shape_here(x, "m n 2")
            return x.sum(axis=(0, 1))
        
        test2(xp.zeros((4, 3, 2)))

        with self.assertRaises(TensorShapeAssertError):
            test2(xp.zeros((4, 3, 1)))
        with self.assertRaises(TensorShapeAssertError):
            test2(xp.zeros((4, 3, 3)))

class TestScalarValues(unittest.TestCase):
    def test_scalar_inputs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor[""], y: ShapedTensor[""]) -> ShapedTensor[""]:
            return x + y
        
        test(xp.asarray(5), xp.asarray(3))
        test(xp.asarray(5.0), xp.asarray(3.2))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.asarray(5), xp.zeros(1))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros(1), xp.asarray(3))

    def test_scalar_outputs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n"]) -> ShapedTensor[""]:
            return x.sum()
        
        test(xp.zeros(5))
        
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["n"]) -> ShapedTensor["1"]:
                return x.sum()
            
            test(xp.zeros(5))

class TestScalarValuesAlias(unittest.TestCase):
    def test_scalar_inputs(self):
        @check_tensor_shapes()
        def test(x: ScalarTensor, y: ScalarTensor) -> ScalarTensor:
            return x + y
        
        test(xp.asarray(5), xp.asarray(3))
        test(xp.asarray(5.0), xp.asarray(3.2))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.asarray(5), xp.zeros(1))

        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros(1), xp.asarray(3))

    def test_scalar_outputs(self):
        @check_tensor_shapes()
        def test1(x: ShapedTensor["n"]) -> ScalarTensor:
            return x.sum()
        
        test1(xp.zeros(5))

        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test2(x: ShapedTensor["n"]) -> ShapedTensor["1"]:
                return x.sum()
            
            test2(xp.zeros(5))

@check_tensor_shapes
class PicklingCompatibilityMyClassNoParentheses:
    def __init__(self, x: ShapedTensor["n"]):
        self.x = x
@check_tensor_shapes()
class PicklingCompatibilityMyClassParentheses:
    def __init__(self, x: ShapedTensor["n"]):
        self.x = x

@check_tensor_shapes()
class PicklingCompatibilityMyNamedTupleNoParentheses(NamedTuple):
    x: ShapedTensor["n"]
@check_tensor_shapes()
class PicklingCompatibilityMyNamedTupleParentheses(NamedTuple):
    x: ShapedTensor["n"]

class TestPicklingCompatibility(unittest.TestCase):
    def test_pickling_of_normal_classes_without_parentheses(self):
        queue = Queue()
        queue.put(PicklingCompatibilityMyClassNoParentheses(xp.zeros(5)))
        queue.get(timeout=1)

    def test_pickling_of_normal_classes_with_parentheses(self):
        queue = Queue()
        queue.put(PicklingCompatibilityMyClassParentheses(xp.zeros(5)))
        queue.get(timeout=1)

    def test_pickling_of_namedtuple_without_parentheses(self):
        queue = Queue()
        queue.put(PicklingCompatibilityMyNamedTupleNoParentheses(x=xp.zeros(5)))
        queue.get(timeout=1)

    def test_pickling_of_namedtuple_with_parentheses(self):
        queue = Queue()
        queue.put(PicklingCompatibilityMyNamedTupleParentheses(x=xp.zeros(5)))
        queue.get(timeout=1)

@unittest.skipIf(sys.platform.startswith("win"), "Skipping torch.compile tests on Windows")
class TestTorchCompile(unittest.TestCase):
    def tearDown(self):
        set_global_check_mode('always')
        return super().tearDown()

    def setUp(self):
        if os.environ["TSA_TEST_LIBRARY"] != "torch":
            self.skipTest("compile only tested for torch")

    def test_torch_compile_disables_check_but_warns(self):
        import torch
        set_global_check_mode('always')

        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))

        # fails uncompiled
        with self.assertRaises(TensorShapeAssertError):
            test(xp.zeros((4, 3, 1)))
        
        compiled_test = torch.compile(test)
        
        # check is skipped after compilation, but warning is raised
        with self.assertWarns(CheckDisabledWarning):
            compiled_test(xp.zeros((4, 3, 1)))

    def test_torch_compile_does_not_warn_if_check_disabled(self):
        import torch
        set_global_check_mode('never')

        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 2"]) -> ShapedTensor["2"]:
            return x.sum(axis=(0, 1))

        compiled_test = torch.compile(test)
        
        # check is skipped after compilation, and no warning is raised
        warnings.filterwarnings("error", category=CheckDisabledWarning)
        compiled_test(xp.zeros((4, 3, 1)))
        warnings.filterwarnings("default", category=CheckDisabledWarning)


        self.assertTrue(True)


class TestNonTensorTupleAnnotations(unittest.TestCase):
    def test_non_tensor_tuple_annotations(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n"], info: tuple[int, int]) -> ShapedTensor["n"]:
            return x
        
        test(x=xp.zeros(10), info=(42, "hello"))
        test(x=xp.zeros(5), info=(7, "world"))

        with self.assertRaises(TensorShapeAssertError):
            test(x=xp.zeros((10, 5)), info=(42, 3.14))


class TestTraceLogging(unittest.TestCase):
    def test_trace_logging_example(self):

        @check_tensor_shapes()
        def f(x: ShapedTensor["a b n"]) -> ShapedTensor["a n"]:
            return xp.sum(x, axis=1)
        
        @check_tensor_shapes()
        def g(x: ShapedTensor["a b 2"]) -> ShapedTensor["a"]:
            y = f(x)
            return y[:, 0] * y[:, 1]

        @check_tensor_shapes()
        def h(x: ShapedTensor["a b n"], n: int = 2) -> tuple[ScalarTensor, ScalarTensor]:
            y = g(x)
            return xp.mean(y), xp.var(y)

        start_trace_recording()
        h(xp.zeros((3, 4, 2)))
        records = stop_trace_recording()
        record_str = trace_records_to_string(records)

        self.assertIn("h (defined at", record_str)
        self.assertIn(", stack index: 0, call index: 0", record_str)
        self.assertIn("|   <int variables> : (int) -> shape () => {'n': 2}", record_str)
        self.assertIn("|   x : (a b n) -> shape (3, 4, 2) => {'n': 2, 'a': 3, 'b': 4}", record_str)
        self.assertIn("|   ", record_str)
        self.assertIn("|   g (defined at", record_str)
        self.assertIn(", stack index: 1, call index: 2", record_str)
        self.assertIn("|   |   x : (a b 2) -> shape (3, 4, 2) => {'a': 3, 'b': 4}", record_str)
        self.assertIn("|   |   ", record_str)
        self.assertIn("|   |   f (defined at", record_str)
        self.assertIn(", stack index: 2, call index: 3", record_str)
        self.assertIn("|   |   |   x : (a b n) -> shape (3, 4, 2) => {'a': 3, 'b': 4, 'n': 2}", record_str)
        self.assertIn("|   |   |   <return> : (a n) -> shape (3, 2) => {'a': 3, 'b': 4, 'n': 2}", record_str)
        self.assertIn("|   |   <return> : (a) -> shape (3,) => {'a': 3, 'b': 4}", record_str)
        self.assertIn("|   <return> : () -> shape () => {'n': 2, 'a': 3, 'b': 4}", record_str)
        self.assertIn("|   <return> : () -> shape () => {'n': 2, 'a': 3, 'b': 4}", record_str)

    def test_tracing_namedtuples(self):
        @check_tensor_shapes()
        class MyInputTuple(NamedTuple):
            p: ShapedTensor["n m"]
            q: ShapedTensor["m 1"]

        @check_tensor_shapes()
        class MyOutputTuple(NamedTuple):
            result: ShapedTensor["n"]

        @check_tensor_shapes()
        def test(x: MyInputTuple) -> MyOutputTuple:
            return MyOutputTuple(result=(x.p @ x.q)[:, 0])

        start_trace_recording()
        test(MyInputTuple(
            p=xp.zeros((5, 4)),
            q=xp.zeros((4, 1))
        ))
        records = stop_trace_recording()
        record_str = trace_records_to_string(records)

        self.assertIn(
            "__new__ (defined at namedtuple_MyInputTuple.MyInputTuple.__new__:-1), stack index: 0, call index: 0\n"
            "|   p : (n m) -> shape (5, 4) => {'n': 5, 'm': 4}\n"
            "|   q : (m 1) -> shape (4, 1) => {'n': 5, 'm': 4}\n"
            "|   \n"
            "|   __new__ (defined at namedtuple_MyOutputTuple.MyOutputTuple.__new__:-1), stack index: 1, call index: 2\n"
            "|   |   result : (n) -> shape (5,) => {'n': 5}",
            record_str
        )
