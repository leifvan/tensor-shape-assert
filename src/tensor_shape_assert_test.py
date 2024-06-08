from typing import Callable, Union
import unittest
import torch
from tensor_shape_assert.wrapper import check_tensor_shapes, ShapedTensor, get_shape_variables, assert_shape_here
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
        


if __name__ == "__main__":
    unittest.main()
