import unittest
import torch
from tensor_shape_assert import check_tensor_shapes, ShapedTensor, IncompatibleShapeError

class Test1DAnnotations(unittest.TestCase):
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
        
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(6)
            test(x=x)
            
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(1, 5)
            test(x=x)
            
        with self.assertRaises(IncompatibleShapeError):
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
        
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(8)
            test(x=x)
            
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(10, 2)
            test(x=x)
            
            
class TestNDAnnotations(unittest.TestCase):
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
        
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(5, 4, 2)
            y = torch.zeros(3, 4, 5)
            test(x=x, y=y)
            
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(5, 4, 3)
            y = torch.zeros(3, 4)
            test(x=x, y=y)
            
        with self.assertRaises(IncompatibleShapeError):
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
        
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(17, 2, 2)
            y = torch.zeros(3, 19, 2)
            test(x=x, y=y)
            
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(17, 3)
            y = torch.zeros(3, 19, 2)
            test(x=x, y=y)
            
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(17, 3, 2)
            y = torch.zeros(3, 2)
            test(x=x, y=y)
            
            
class TestArbitraryBatchDimAnnotations(unittest.TestCase):
    def test_constant_nd_arbitrary_batch_input_shape_checked(self):
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
        
    def test_constant_nd_arbitrary_batch_input_shape_error(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["... 5 1"], y: ShapedTensor["... 1 3 2"]) -> ShapedTensor["... 2"]:
            return torch.zeros(6, 5, 4, 3, 1)
        
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(1, 5, 2)
            y = torch.zeros(1, 1, 3)
            test(x=x, y=y)
        
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(15, 5, 1)
            y = torch.zeros(20, 1, 3)
            test(x=x, y=y)
            
        with self.assertRaises(IncompatibleShapeError):
            x = torch.zeros(15, 5, 1)
            y = torch.zeros(20, 1, 3, 2)
            test(x=x, y=y)
        
    def test_invalid_arbitrary_shape_annotation(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["5 ..."], y: ShapedTensor["1 3"]) -> ShapedTensor["2"]:
            return torch.zeros(2)
        
        with self.assertRaises(ValueError):
            test(x=torch.zeros(5, 1), y=torch.zeros(1, 3))
        
class TestNamedBatchDimensions(unittest.TestCase):
    def test_correct_1_batch_dimension_equal_length(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3 2"]
        ) -> ShapedTensor["...B 2 2"]:
            return x @ y
        
        test(x=torch.zeros(12, 2, 3), y=torch.zeros(12, 3, 2))
        self.assertTrue(True)

    def test_correct_1_batch_dimension_x_longer_y(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2 3"],
            y: ShapedTensor["...B 3"]
        ) -> ShapedTensor["...B 2 3"]:
            return x * y[..., None, :]
        
        test(x=torch.zeros(12, 2, 3), y=torch.zeros(12, 3))
        self.assertTrue(True)

    def test_correct_1_batch_dimension_y_longer_x(self):
        @check_tensor_shapes()
        def test(
            x: ShapedTensor["...B 2"],
            y: ShapedTensor["...B 2 3"]
        ) -> ShapedTensor["...B 2 3"]:
            return x[..., None] * y
        
        test(x=torch.zeros(12, 2), y=torch.zeros(12, 2, 3))
        self.assertTrue(True)

    def test_correct_2_batch_dimensions(self):
        @check_tensor_shapes()
        def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
            return x @ y
        test(x=torch.zeros(12, 3, 2, 3), y=torch.zeros(12, 3, 3, 2))
        self.assertTrue(True)

    def test_wrong_length_batch_dimensions_input(self):
        with self.assertRaises(IncompatibleShapeError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return x @ y
            test(x=torch.zeros(12, 2, 3), y=torch.zeros(12, 3, 2, 3))

    def test_wrong_length_batch_dimensions_output(self):
        with self.assertRaises(IncompatibleShapeError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return (x @ y)[0]
            test(x=torch.zeros(12, 3, 2, 3), y=torch.zeros(12, 3, 2, 3))

    def test_wrong_size_batch_dimensions_input(self):
        with self.assertRaises(IncompatibleShapeError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return x @ y
            test(x=torch.zeros(12, 4, 2, 3), y=torch.zeros(12, 3, 2, 3))

    def test_wrong_size_batch_dimensions_output(self):
        with self.assertRaises(IncompatibleShapeError):
            @check_tensor_shapes()
            def test(x: ShapedTensor["...B 2 3"], y: ShapedTensor["...B 3 2"]) -> ShapedTensor["...B 2 2"]:
                return (x @ y)[:5]
            test(x=torch.zeros(12, 4, 2, 3), y=torch.zeros(12, 4, 2, 3))
        
class TestMisc(unittest.TestCase):
    def test_instantiating(self):
        with self.assertRaises(RuntimeError):
            ShapedTensor()
            
    def test_keyword_only(self):
        with self.assertWarns(RuntimeWarning):
            @check_tensor_shapes()
            def test(x: ShapedTensor["1"]):
                return x
            
            test(torch.zeros(1))



if __name__ == "__main__":
    unittest.main()
