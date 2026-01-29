import torch
from src.tensor_shape_assert import (
    check_tensor_shapes, ShapedTensor, ScalarTensor, start_trace_recording,
    stop_trace_recording, trace_records_to_string
)
from typing import NamedTuple

if __name__ == "__main__":

    @check_tensor_shapes()
    class Result(NamedTuple):
        mean: ScalarTensor
        var: ScalarTensor

    @check_tensor_shapes()
    def f(x: ShapedTensor["a b n"]) -> ShapedTensor["a n"]:
        return x.sum(dim=1)
    
    @check_tensor_shapes()
    def g(x: ShapedTensor["a b 2"]) -> ShapedTensor["a"]:
        y = f(x)
        return y[:, 0] * y[:, 1]

    @check_tensor_shapes()
    def h(x: ShapedTensor["a b n"], n: int = 2) -> Result:
        y = g(x)
        return Result(mean=y.mean(), var=y.var())

    start_trace_recording()
    h(torch.randn(3, 4, 2))

    records = stop_trace_recording()
    print(trace_records_to_string(records))

    @check_tensor_shapes()
    def rec(x: ShapedTensor["b m m"], n: int) -> tuple[ShapedTensor["b m m"], int]:
        if n == 0:
            return x, n
        else:
            return rec(x @ x, n - 1)

    start_trace_recording()
    rec(torch.randn(2, 3, 3), 10)
    records = stop_trace_recording()
    print(trace_records_to_string(records))

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
        p=torch.zeros((5, 4)),
        q=torch.zeros((4, 1))
    ))
    records = stop_trace_recording()
    record_str = trace_records_to_string(records)
    print(record_str)