from typing import Callable, NamedTuple
import unittest
import torch
from tensor_shape_assert.wrapper import check_tensor_shapes, ShapedTensor, get_shape_variables, assert_shape_here, set_global_check_mode, ScalarTensor
from tensor_shape_assert.utils import TensorShapeAssertError
from tensor_shape_assert.wrapper import NoVariableContextExistsError, VariableConstraintError


class Test1DAnnotationsKeyword(unittest.TestCase):
    def test_constant_1d_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5"]) -> ShapedTensor["2"]:
            return x[:2]
        
        x = torch.zeros(5)
        self.assertTupleEqual(test(x=x).shape, (2,))

    def test_constant_1d_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5"]) -> ShapedTensor["2"]:
            return x[:2]
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(6)
            test(x=x)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(1, 5)
            test(x=x)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(5, 1)
            test(x=x)
            
    def test_variable_1d_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["b"]:
            return x + 1
        
        x = torch.zeros(8)
        self.assertTupleEqual(test(x=x).shape, x.shape)
        self.assertTrue(torch.all(x != test(x=x)))
        
    def test_variable_1d_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["b"]:
            return x[:2]
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            test(x=x)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(10, 2)
            test(x=x)

    def test_no_output_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]):
            return x + 1
        
        x = torch.zeros(8)
        self.assertTupleEqual(test(x=x).shape, x.shape)
        self.assertTrue(torch.all(x != test(x=x)))

    def test_no_output_annotation_with_tuple(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]):
            return (x + 1, x + 2)
        
        x = torch.zeros(8)
        self.assertTupleEqual(test(x=x)[0].shape, x.shape)
        self.assertTrue(torch.all(x != test(x=x)[0]))

    def test_wrong_output_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["1"]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            self.assertTupleEqual(test(x=x).shape, x.shape)
            self.assertTrue(torch.all(x != test(x=x)))

    def test_wrong_output_annotation_one_item(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1"], ShapedTensor["2"]]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            test(x=x)

    def test_wrong_output_annotation_multiple_items(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1"], ShapedTensor["2"], ShapedTensor["3"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            test(x=x)

    def test_wrong_output_annotation_more_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> ShapedTensor["1 b 2 3"]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            self.assertTupleEqual(test(x=x).shape, x.shape)
            self.assertTrue(torch.all(x != test(x=x)))

    def test_wrong_output_annotation_less_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 1 1 2"]) -> ShapedTensor["1"]:
            return x + 1
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8, 1, 1, 2)
            self.assertTupleEqual(test(x=x).shape, x.shape)
            self.assertTrue(torch.all(x != test(x=x)))


    def test_wrong_output_annotation_tuple(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1"], ShapedTensor["b"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            self.assertTupleEqual(test(x=x)[0].shape, x.shape)
            self.assertTrue(torch.all(x != test(x=x)[0]))

    def test_wrong_output_annotation_tuple_more_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b"]) -> tuple[ShapedTensor["1 b"], ShapedTensor["2 b"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8)
            self.assertTupleEqual(test(x=x)[0].shape, x.shape)
            self.assertTrue(torch.all(x != test(x=x)[0]))

    def test_wrong_output_annotation_tuple_less_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 1 2 3"]) -> tuple[ShapedTensor["b"], ShapedTensor["b"]]:
            return (x + 1, x + 2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(8, 1, 2, 3)
            self.assertTupleEqual(test(x=x)[0].shape, x.shape)
            self.assertTrue(torch.all(x != test(x=x)[0]))
            
class TestNDAnnotationsKeyword(unittest.TestCase):
    def test_constant_nd_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5 1"], y: ShapedTensor["1 3"]) -> ShapedTensor["2"]:
            return x[:2, 0] + y[0, 1:]
        
        x = torch.zeros(5, 1)
        y = torch.zeros(1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (2,))
        
    def test_constant_nd_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5 4 3"], y: ShapedTensor["3 4 5"]) -> ShapedTensor["3"]:
            return x[1, 2, :] + y[:, 1, 2]
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(5, 4, 2)
            y = torch.zeros(3, 4, 5)
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(5, 4, 3)
            y = torch.zeros(3, 4)
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(5)
            y = torch.zeros(3, 4, 5)
            test(x=x, y=y)
            
    def test_variable_nd_input_shape_checked(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 3 2"], y: ShapedTensor["3 c 2"]) -> ShapedTensor["3 c"]:
            return (x[:1, :1] + y).sum(dim=2)
        
        x = torch.zeros(17, 3, 2)
        y = torch.zeros(3, 19, 2)
        self.assertTupleEqual(test(x=x, y=y).shape, (3, 19))
        
        x = torch.zeros(1, 3, 2)
        y = torch.zeros(3, 19, 2)
        self.assertTupleEqual(test(x=x, y=y).shape, (3, 19))
        
        x = torch.zeros(17, 3, 2)
        y = torch.zeros(3, 12, 2)
        self.assertTupleEqual(test(x=x, y=y).shape, (3, 12))
        
    def test_variable_nd_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["b 3 2"], y: ShapedTensor["3 c 2"]) -> ShapedTensor["3 c"]:
            return (x[:1, :1] + y).sum(dim=2)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(17, 2, 2)
            y = torch.zeros(3, 19, 2)
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(17, 3)
            y = torch.zeros(3, 19, 2)
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(17, 3, 2)
            y = torch.zeros(3, 2)
            test(x=x, y=y)
            
            
class TestArbitraryBatchDimAnnotationsKeyword(unittest.TestCase):
    def test_constant_nd_arbitrary_batch_input_shape_checked_front(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["... 5 1"], y: ShapedTensor["... 1 3"]) -> ShapedTensor["... 2"]:
            return x[..., :2, 0] + y[..., 0, 1:]
        
        x = torch.zeros(5, 1)
        y = torch.zeros(1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (2,))
        
        x = torch.zeros(1, 5, 1)
        y = torch.zeros(1, 1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (1, 2))
        
        x = torch.zeros(32, 5, 1)
        y = torch.zeros(32, 1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (32, 2))
        
        x = torch.zeros(9, 8, 7, 5, 1)
        y = torch.zeros(9, 8, 7, 1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (9, 8, 7, 2))

    def test_constant_nd_arbitrary_batch_input_shape_checked_middle(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5 ... 1"], y: ShapedTensor["1 ... 3"]) -> ShapedTensor["... 2"]:
            return x[4, ..., (0, 0)] + y[0, ..., 1:]
        
        x = torch.zeros(5, 1)
        y = torch.zeros(1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (2,))
        
        x = torch.zeros(5, 1, 1)
        y = torch.zeros(1, 1, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (1, 2))
        
        x = torch.zeros(5, 32, 1)
        y = torch.zeros(1, 32, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (32, 2))
        
        x = torch.zeros(5, 9, 8, 7, 1)
        y = torch.zeros(1, 9, 8, 7, 3)
        self.assertTupleEqual(test(x=x, y=y).shape, (9, 8, 7, 2))
        
    def test_constant_nd_arbitrary_batch_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["... 5 1"], y: ShapedTensor["... 1 3 2"]) -> ShapedTensor["... 2"]:
            return torch.zeros(6, 5, 4, 3, 1)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(1, 5, 2)
            y = torch.zeros(1, 1, 3)
            test(x=x, y=y)
        
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(15, 5, 1)
            y = torch.zeros(20, 1, 3)
            test(x=x, y=y)
            
        with self.assertRaises(TensorShapeAssertError):
            x = torch.zeros(15, 5, 1)
            y = torch.zeros(20, 1, 3, 2)
            test(x=x, y=y)


class TestNamedBatchDimensions(unittest.TestCase):
    def test_correct_1_batch_dimension_equal_length(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3 2"]
        ) -> ShapedTensor["...B 2 2"]:
            return x @ y
        
        test(x=torch.zeros(12, 2, 3), y=torch.zeros(12, 3, 2))

    def test_correct_1_batch_dimension_x_longer_y(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3"]
        ) -> ShapedTensor["...B 2 3"]:
            return x * y[..., None, :]
        
        test(x=torch.zeros(12, 2, 3), y=torch.zeros(12, 3))

    def test_correct_1_batch_dimension_y_longer_x(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2"],
            y: ShapedTensor["...B 2 3"]
        ) -> ShapedTensor["...B 2 3"]:
            return x[..., None] * y
        
        test(x=torch.zeros(12, 2), y=torch.zeros(12, 2, 3))

    def test_correct_2_batch_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
            return x @ y
        test(x=torch.zeros(12, 3, 2, 3), y=torch.zeros(12, 3, 3, 2))

    def test_correct_2_batch_dimensions_tuple_output(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3 2"]
        ) -> tuple[
            ShapedTensor["...B 2 2"],
            ShapedTensor["...B"]
        ]:
            return x @ y, (x @ y).sum(dim=(-1, -2))
        
        test(x=torch.zeros(12, 3, 2, 3), y=torch.zeros(12, 3, 3, 2))

    def test_wrong_length_batch_dimensions_input(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return x @ y
            test(x=torch.zeros(12, 2, 3), y=torch.zeros(12, 3, 2, 3))

    def test_wrong_length_batch_dimensions_output(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return (x @ y)[0]
            test(x=torch.zeros(12, 3, 2, 3), y=torch.zeros(12, 3, 2, 3))

    def test_wrong_size_batch_dimensions_input(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return x @ y
            test(x=torch.zeros(12, 4, 2, 3), y=torch.zeros(12, 3, 2, 3))

    def test_wrong_size_batch_dimensions_output(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return (x @ y)[:5]
            test(x=torch.zeros(12, 4, 2, 3), y=torch.zeros(12, 4, 2, 3))

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
        
        Test().my_method(x=torch.zeros(17, 3), y=torch.zeros(17))
        
        with self.assertRaises(TensorShapeAssertError):
            Test().my_method(x=torch.zeros(17, 4), y=torch.zeros(17))

    def test_constructor_annotation(self):
        class Test:
            @check_tensor_shapes()
            def __init__(self, x: ShapedTensor["b 3"], y: ShapedTensor["b"]):
                self.z = x + y[:, None]
        
        Test(x=torch.zeros(17, 3), y=torch.zeros(17))

        with self.assertRaises(TensorShapeAssertError):
            Test(x=torch.zeros(17, 4), y=torch.zeros(17))

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
        
        SubTest().my_method(x=torch.zeros(17, 3), y=torch.zeros(17))

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
                return super().my_method(x=x, y=y).sum(dim=1)
        
        SubTest().my_method(x=torch.zeros(17, 3), y=torch.zeros(17))


class TestTensorDescriptor(unittest.TestCase):
    @check_tensor_shapes()
    def test_multiple_whitespaces_ignored(self):
        def test(x: ShapedTensor["a   b c  "]) -> ShapedTensor[" b   "]:
            return x.mean(dim=0)[:, 0]
        test(x=torch.zeros(3, 4, 5))
    
    def test_commas_ignored(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a,b,c"]) -> ShapedTensor["b,,"]:
            return x.mean(dim=0)[:, 0]
        test(x=torch.zeros(3, 4, 5))

    def test_parentheses_ignored(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b[c"]) -> ShapedTensor["b]"]:
            return x.mean(dim=0)[:, 0]
        test(x=torch.zeros(3, 4, 5))

        
class TestMisc(unittest.TestCase):
    def test_instantiating(self):
        with self.assertRaises(RuntimeError):
            ShapedTensor()

    def test_misc_annotations_ignored(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["1"]) -> str:
            return "hi"
        
        test(x=torch.zeros(1))

        @check_tensor_shapes()
        def test(x: ShapedTensor["1"]) -> tuple[int, int, str]:
            return 1, 2, "hi"
        
        test(x=torch.zeros(1))

        @check_tensor_shapes()
        def test(x: ShapedTensor["1"]) -> list[str]:
            return ["hi", "bye"]
        
        test(x=torch.zeros(1))

        @check_tensor_shapes()
        def test(x: ShapedTensor["1"]) -> Callable[[str, str], int]:
            def _test(a: str, b: str) -> int:
                return len(a) + len(b)
            return _test
        
        test(x=torch.zeros(1))

    def test_helpful_error_message_when_brackets_missing(self):
        
        with self.assertRaises(TensorShapeAssertError) as cm:
            @check_tensor_shapes
            def test(x: ShapedTensor["a"]) -> ShapedTensor["b"]:
                return x[1:]
            
            test(x=torch.zeros(5))
        
        self.assertIn("Maybe you forgot brackets", str(cm.exception))

    def test_union_type_error(self):
        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["a"] | str):
                return x
            
            test(x=torch.zeros(5))



class TestGetVariableValuesFromCurrentContext(unittest.TestCase):
    def test_single_context(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b"]) -> ShapedTensor["a"]:
            a, b = get_shape_variables("a b")
            self.assertTupleEqual(x.shape, (a, b))
            return x.sum(dim=1)
        
        test(x=torch.ones(5, 6))

    def test_nested_context(self):
        x_input = torch.ones(5, 6, 7)

        @check_tensor_shapes()
        def sum_plus_one(x: ShapedTensor["a b"]) -> ShapedTensor["b"]:
            a, b = get_shape_variables("a b")
            self.assertTupleEqual(x_input.shape[1:], (a, b))
            return x.sum(dim=0) + 1

        @check_tensor_shapes()
        def test(x: ShapedTensor["a b c"]) -> ShapedTensor["c"]:
            a, b, c = get_shape_variables("a b c")
            self.assertTupleEqual(x_input.shape, (a, b, c))
            y = sum_plus_one(x=x[0])
            return y
        
        test(x=x_input)

    def test_error_on_no_context(self):
        with self.assertRaises(NoVariableContextExistsError):
            a, b = get_shape_variables("a b")
            print(a, b)

    def test_unknown_variable_is_none(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b"]) -> ShapedTensor["a"]:
            c = get_shape_variables("c")
            self.assertIsNone(c)
            return x.sum(dim=1)
        
        test(x=torch.ones(5, 6))

    def test_single_context(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a b"]) -> ShapedTensor["a"]:
            a, b = get_shape_variables(" a  ] [[b    ")
            self.assertTupleEqual(x.shape, (a, b))
            return x.sum(dim=1)
        
        test(x=torch.ones(5, 6))


class TestPositionalAndMixedArguments(unittest.TestCase):
    def test_positional_only(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["c 3"], y: ShapedTensor["3 c"]) -> ShapedTensor["c c"]:
            return x @ y
        test(torch.zeros(5, 3), torch.ones(3, 5))

    def test_mixed_positional_and_kwargs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["c 3"], y: ShapedTensor["3 c"]) -> ShapedTensor["c c"]:
            return x @ y
        test(torch.zeros(5, 3), y=torch.ones(3, 5))


class TestArbitrarilyNestedInputs(unittest.TestCase):
    def test_flat_input_tuples(self):
        @check_tensor_shapes()
        def test(x: tuple[
                ShapedTensor["a 2"],
                ShapedTensor["b 2"],
        ], *args) -> ShapedTensor["a"]:
            return (x[0] * x[1][0][0]).sum(dim=1)
        
        test((
            torch.zeros(10, 2),
            torch.zeros(9, 2)
        ))

        test((
            torch.zeros(28, 2),
            torch.zeros(49, 2)
        ))

        test([
            torch.zeros(10, 2),
            torch.zeros(9, 2)
        ])

        with self.assertRaises(TensorShapeAssertError):
            test((
                torch.zeros(10, 3),
                torch.zeros(9, 2)
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                torch.zeros(10, 2),
                torch.zeros(2)
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                torch.zeros(10, 2),
                torch.zeros(3, 2),
                torch.zeros(2, 2)
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                torch.zeros(10, 2),
            ))

        with self.assertRaises(TensorShapeAssertError):
            test(
                torch.zeros(10, 2),
                torch.zeros(9, 2)
            )

        with self.assertRaises(AttributeError):
            test((
                (torch.zeros(10, 2),),
                (torch.zeros(10, 2),),
            ))
             
    def test_complex_input_tuples(self):
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
                (torch.zeros(4), torch.zeros(3)),
                torch.zeros(4, 2)
            ),
            torch.zeros(3, 2)
        ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                (
                    (torch.zeros(4), torch.zeros(4)),
                    torch.zeros(4, 2)
                ),
                torch.zeros(3, 2)
            ))

        with self.assertRaises(TensorShapeAssertError):
            test((
                (
                    (torch.zeros(4), torch.zeros(3)),
                    torch.zeros(4, 3)
                ),
                torch.zeros(3, 2)
            ))
        
class TestLocalShapeChecking(unittest.TestCase):
    def test_local_shape_check_positive(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"], y: ShapedTensor["...B 3"]) -> ShapedTensor["...B"]:
            z = y[..., 0] + x[0]
            assert_shape_here(z, "...B")
            return z

        test(x=torch.zeros(2, 1), y=torch.zeros(5, 4, 3))

    def test_local_shape_check_negative(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"], y: ShapedTensor["...B 3"]) -> ShapedTensor["...B"]:
            z = y[..., 0] + x[0]
            assert_shape_here(z, "a")
            return z

        with self.assertRaises(TensorShapeAssertError):
            test(x=torch.zeros(2, 1), y=torch.zeros(5, 4, 3))

    def test_local_shape_check_updates_context(self):
        @check_tensor_shapes()
        def test(
                x: ShapedTensor["a"],
                y: ShapedTensor["b"],
                c: int | None
        ) -> ShapedTensor["c"]:
            z = x[None, :] * y[:, None]
            k = torch.zeros(1, 1, 5)
            f = z[:, :, None] * k

            if c is not None:
                assert_shape_here((c,), "c")

            return f.view(-1, 5).sum(dim=1)
        
        test(torch.zeros(6), torch.zeros(7), c=42)
        test(torch.zeros(6), torch.zeros(7), c=None)

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(6), torch.zeros(7), c=41)

        
class TestOptionalShapeAnnotation(unittest.TestCase):
    def test_optional_input_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"] | None) -> ShapedTensor["1"]:
            if x is not None:
                return x[0]
            else:
                return torch.zeros(1)
        
        test(torch.zeros(2, 1))
        test(None)

    def test_warn_optional_output_tuple(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["a 1"] | None) -> tuple[ShapedTensor["1"]] | None:
            if x is not None:
                return x[0]
            else:
                return (torch.zeros(1), )
        
        with self.assertWarns(RuntimeWarning):
            test(torch.zeros(2, 1))        

class TestVariableConstraints(unittest.TestCase):
    def test_lambda_constraints(self):
        @check_tensor_shapes(constraints=[lambda v: v['a'] + v['b']==5])
        def test(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> float:
            return a[0] + b[0]
        
        test(torch.zeros(3), torch.zeros(2))
        test(torch.zeros(1), torch.zeros(4))

        with self.assertRaises(VariableConstraintError):
            test(torch.zeros(3), torch.zeros(3))

    def test_str_expression_constraints(self):
        @check_tensor_shapes(constraints=["a + b = c"])
        def test(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["c"]:
            return torch.cat([a, b], dim=0)
        
        test(torch.zeros(3), torch.zeros(2))
        test(torch.zeros(1), torch.zeros(4))

        @check_tensor_shapes(constraints=["a + b = c"])
        def test(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["c"]:
            return torch.cat([a, b, torch.zeros(1)], dim=0)
        
        with self.assertRaises(VariableConstraintError):
            test(torch.zeros(3), torch.zeros(3))

    def test_autogen_constraints(self):
        @check_tensor_shapes(experimental_enable_autogen_constraints=True)
        def test(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["a+b"]:
            return torch.cat([a, b], dim=0)
        
        test(torch.zeros(3), torch.zeros(2))
        test(torch.zeros(1), torch.zeros(4))

        @check_tensor_shapes(experimental_enable_autogen_constraints=True)
        def test(a: ShapedTensor["a"], b: ShapedTensor["b"]) -> ShapedTensor["a+b"]:
            return torch.cat([a, b, torch.zeros(1)], dim=0)
        
        with self.assertRaises(VariableConstraintError):
            test(torch.zeros(3), torch.zeros(3))

class TestIntToVariables(unittest.TestCase):
    def test_int_to_variable(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3"], m: int) -> ShapedTensor["m"]:
            return x.sum(dim=0)[:, 0]

        test(torch.zeros(5, 4, 3), m=4)
        test(torch.zeros(5, 2, 3), m=2)

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(5, 4, 3), m=3)        

    def test_int_to_variable_deactivated(self):
        @check_tensor_shapes(ints_to_variables=False)
        def test(x: ShapedTensor["n m 3"], m: int) -> ShapedTensor["m"]:
            return x.sum(dim=0)[:, 0]

        test(torch.zeros(5, 4, 3), m=4)
        test(torch.zeros(5, 2, 3), m=2)
        test(torch.zeros(5, 4, 3), m=3)

    def test_strict_int_comparison(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3"], m: bool) -> ShapedTensor["m"]:
            return x.sum(dim=0)[:, 0]
        
        test(torch.zeros(5, 4, 3), m=True)

class TestDtypeAnnotationTorch(unittest.TestCase):
    def test_int_types(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", torch.int16]) -> ShapedTensor["m", torch.int32]:
            return x.sum(dim=2).median(dim=0)[0].to(torch.int32)
        
        test(torch.zeros(1, 2, 3, dtype=torch.int16))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, dtype=torch.int32))

    def test_float_types(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", torch.float64]) -> ShapedTensor["m", torch.float32]:
            return x.sum(dim=2).median(dim=0)[0].to(torch.float32)
        
        test(torch.zeros(1, 2, 3, dtype=torch.float64))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, dtype=torch.float16))

    def test_complex_type(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", torch.complex128]) -> ShapedTensor["m", torch.float16]:
            return x.sum(dim=(0, 2)).real.to(torch.float16)
        
        test(torch.zeros(1, 2, 3, dtype=torch.complex128))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, dtype=torch.float64))

    def test_ignored_type(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", None], rt) -> ShapedTensor["m", torch.float16]:
            return x.sum(dim=(0, 2)).real.to(rt)
        
        test(torch.zeros(1, 2, 3, dtype=torch.complex64), rt=torch.float16)

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, dtype=torch.complex64), rt=torch.float32)

class TestDeviceAnnotationTorch(unittest.TestCase):
    def test_device_cpu(self):
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping device test...")
            return

        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", None, 'cpu']) -> ShapedTensor["m", None, 'cpu']:
            return x.sum(dim=2).median(dim=0)[0].to(torch.int32)
        
        test(torch.zeros(1, 2, 3, device='cpu'))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, device='cuda:0'))

    def test_device_single_gpu(self):
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping device test...")
            return

        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", None, 'cuda:0']) -> ShapedTensor["m", None, 'cpu']:
            return x.sum(dim=2).median(dim=0)[0].to('cpu')
        
        test(torch.zeros(1, 2, 3, device='cuda:0'))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, device='cpu'))

    def test_device_multi_gpu(self):
        
        if torch.cuda.device_count() < 2:
            print("Required amount of CUDA devices not available, skipping device test...")
            return

        @check_tensor_shapes()
        def test(x: ShapedTensor["n m 3", None, 'cuda:0']) -> ShapedTensor["m", None, 'cuda:1']:
            return x.sum(dim=2).median(dim=0)[0].to('cuda:1')
        
        test(torch.zeros(1, 2, 3, device='cuda:0'))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1, 2, 3, device='cpu'))


class TestNamedTupleSupport(unittest.TestCase):
    def test_named_tuple(self):
        
        @check_tensor_shapes()
        class MyTuple(NamedTuple):
            p: ShapedTensor["n m 3"]
            q: ShapedTensor["n 1 3"]
            num: int

        MyTuple(p=torch.zeros(5, 4, 3), q=torch.zeros(5, 1, 3), num=5)
        MyTuple(torch.zeros(10, 1, 3), torch.zeros(10, 1, 3), num=8)

        with self.assertRaises(TensorShapeAssertError):
            MyTuple(p=torch.zeros(5, 4, 3), q=torch.zeros(5, 2, 3), num=5)


class TestCheckMode(unittest.TestCase):
    def tearDown(self) -> None:
        # reset it here just to be safe
        set_global_check_mode('always')
        return super().tearDown()

    def test_always_checked_by_default(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))
        
        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 1))

    def test_once_checked_global_ignores_errors(self):
        set_global_check_mode('once')

        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))
        test(torch.zeros(4, 3, 1))
        test(torch.zeros(4, 3, 3))

    def test_once_checked_global_detects_first_error(self):
        set_global_check_mode('once')

        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 1))

        set_global_check_mode('always')

    def test_never_global_detects_no_errors(self):
        set_global_check_mode('never')

        @check_tensor_shapes()
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))
        test(torch.zeros(4, 3, 1))
        test(torch.zeros(4, 3, 3))

        set_global_check_mode('always')

    def test_never_global_ignores_assert_shape_here(self):
        set_global_check_mode('never')

        @check_tensor_shapes()
        def test(x: torch.Tensor) -> ShapedTensor["2"]:
            assert_shape_here(x, "m n 2")
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))
        test(torch.zeros(4, 3, 1))
        test(torch.zeros(4, 3, 3))

        set_global_check_mode('always')

    def test_local_always_overrides_global_never(self):
        set_global_check_mode('never')

        @check_tensor_shapes(check_mode='always')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        @check_tensor_shapes()
        def test2(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 1))
        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 3))

        test2(torch.zeros(4, 3, 1))

        set_global_check_mode('always')

    def test_local_always_overrides_global_once(self):
        set_global_check_mode('once')

        @check_tensor_shapes(check_mode='always')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        @check_tensor_shapes()
        def test2(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 1))
        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 3))

        with self.assertRaises(TensorShapeAssertError):
            test2(torch.zeros(4, 3, 1))

        test2(torch.zeros(4, 3, 5))

        set_global_check_mode('always')

    def test_local_never_overrides_global_always(self):

        set_global_check_mode('always')

        @check_tensor_shapes(check_mode='never')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        @check_tensor_shapes()
        def test2(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        test(torch.zeros(4, 3, 2))
        test(torch.zeros(4, 3, 1))
        test(torch.zeros(4, 3, 3))

        with self.assertRaises(TensorShapeAssertError):
            test2(torch.zeros(4, 3, 1))
        with self.assertRaises(TensorShapeAssertError):
            test2(torch.zeros(4, 3, 3))

    def test_local_once_overrides_global_always(self):

        set_global_check_mode('always')

        @check_tensor_shapes(check_mode='once')
        def test(x: ShapedTensor["m n 2"]) -> ShapedTensor["2"]:
            return x.sum(dim=(0, 1))
        
        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(4, 3, 1))

        test(torch.zeros(4, 3, 5))
        test(torch.zeros(4, 3, 3))

class TestScalarValues(unittest.TestCase):
    def test_scalar_inputs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor[""], y: ShapedTensor[""]) -> ShapedTensor[""]:
            return x + y
        
        test(torch.tensor(5), torch.tensor(3))
        test(torch.tensor(5.0), torch.tensor(3.2))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.tensor(5), torch.zeros(1))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1), torch.tensor(3))

    def test_scalar_outputs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n"]) -> ShapedTensor[""]:
            return x.sum()
        
        self.assertTrue(torch.equal(test(torch.zeros(5)), torch.tensor(0)))
        self.assertTrue(torch.equal(test(torch.ones(5)), torch.tensor(5)))

        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["n"]) -> ShapedTensor["1"]:
                return x.sum()
            
            test(torch.zeros(5))

class TestScalarValuesAlias(unittest.TestCase):
    def test_scalar_inputs(self):
        @check_tensor_shapes()
        def test(x: ScalarTensor, y: ScalarTensor) -> ScalarTensor:
            return x + y
        
        test(torch.tensor(5), torch.tensor(3))
        test(torch.tensor(5.0), torch.tensor(3.2))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.tensor(5), torch.zeros(1))

        with self.assertRaises(TensorShapeAssertError):
            test(torch.zeros(1), torch.tensor(3))

    def test_scalar_outputs(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["n"]) -> ScalarTensor:
            return x.sum()
        
        self.assertTrue(torch.equal(test(torch.zeros(5)), torch.tensor(0)))
        self.assertTrue(torch.equal(test(torch.ones(5)), torch.tensor(5)))

        with self.assertRaises(TensorShapeAssertError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["n"]) -> ShapedTensor["1"]:
                return x.sum()
            
            test(torch.zeros(5))

if __name__ == "__main__":
    unittest.main()
