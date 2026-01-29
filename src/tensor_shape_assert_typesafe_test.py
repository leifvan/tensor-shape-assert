import unittest
import os
from typing import TYPE_CHECKING
from tensor_shape_assert.types import (
    ShapedLiteral,
    ShapedNumpyLiteral,
    ShapedTorchLiteral,
    ShapedTensor
)
from tensor_shape_assert.wrapper import (
    check_tensor_shapes,
)
from tensor_shape_assert.utils import TensorShapeAssertError
from typing_extensions import Literal as L

# read library to be used from env
lib = os.environ["TSA_TEST_LIBRARY"]


class TestAnnotationWithLiterals(unittest.TestCase):
    def test_literal_annotation_torch(self):
        if lib != "torch":
            self.skipTest("Skipping torch-specific test")

        import torch
        
        @check_tensor_shapes()
        def test(
                x: ShapedLiteral[torch.Tensor, L["3 a 5"]]
        ) -> ShapedLiteral[torch.Tensor, L["a 5"]]:
            return torch.sum(x, dim=0)

        test(x=torch.zeros((3, 4, 5)))
        with self.assertRaises(TensorShapeAssertError):
            test(x=torch.zeros((2, 4, 5)))

    def test_literal_annotation_torch_alias(self):
        if lib != "torch":
            self.skipTest("Skipping torch-specific test")

        import torch
        
        @check_tensor_shapes()
        def test(
                x: ShapedTorchLiteral[L["3 a 5"]]
        ) -> ShapedTorchLiteral[L["a 5"]]:
            return torch.sum(x, dim=0)

        test(x=torch.zeros((3, 4, 5)))
        with self.assertRaises(TensorShapeAssertError):
            test(x=torch.zeros((2, 4, 5)))

    def test_literal_annotation_numpy(self):
        if lib != "numpy":
            self.skipTest("Skipping numpy-specific test")

        import numpy as np
        
        @check_tensor_shapes()
        def test(
                x: ShapedLiteral[np.ndarray, L["3 a 5"]]
        ) -> ShapedLiteral[np.ndarray, L["a 5"]]:
            return np.sum(x, axis=0)

        test(x=np.zeros((3, 4, 5)))
        with self.assertRaises(TensorShapeAssertError):
            test(x=np.zeros((2, 4, 5)))


    def test_literal_annotation_numpy_alias(self):
        if lib != "numpy":
            self.skipTest("Skipping numpy-specific test")

        import numpy as np
        
        @check_tensor_shapes()
        def test(
                x: ShapedNumpyLiteral[L["3 a 5"]]
        ) -> ShapedNumpyLiteral[L["a 5"]]:
            return np.sum(x, axis=0)

        test(x=np.zeros((3, 4, 5)))
        with self.assertRaises(TensorShapeAssertError):
            test(x=np.zeros((2, 4, 5)))



if __name__ == "__main__":
    unittest.main()
