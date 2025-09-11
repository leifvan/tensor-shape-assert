from src.tensor_shape_assert import ShapedTensor, check_tensor_shapes, set_global_check_mode
import torch
from time import time
from typing import NamedTuple


def benchmark(f, num_runs, num_additonal_args):
    start_time = time()

    add = [torch.zeros(10, 5) for _ in range(num_additonal_args)]

    for _ in range(num_runs):
        f(
             torch.zeros((10, 20, 30)),
             torch.zeros((20, 2)),
             *add
        )
    return (time() - start_time) / num_runs

def func(x: ShapedTensor["a b c"], y: ShapedTensor["b d"], *args: tuple[ShapedTensor["10 5"], ...]) -> ShapedTensor["a d"]:
        k = sum(args)
        z = x[..., None] * y[None, :, None]  # a b c d
        return z.sum(dim=(1, 2))  # a d

if __name__ == "__main__":

    num_runs = 10000
    for num_add_args in (0, 10, 100):

        func_always = check_tensor_shapes(check_mode="always")(func)
        func_once = check_tensor_shapes(check_mode="once")(func)
        func_never = check_tensor_shapes(check_mode="never")(func)

        # full shape checking

        duration_with = benchmark(func_always, num_runs=num_runs, num_additonal_args=num_add_args)

        # check once
        
        set_global_check_mode("once")

        duration_global_check_once = benchmark(func_once, num_runs=num_runs, num_additonal_args=num_add_args)

        # disabled global checking
        
        set_global_check_mode("never")

        duration_global_check_never = benchmark(func_never, num_runs=num_runs, num_additonal_args=num_add_args)

        # no annotations

        duration_not_annotated = benchmark(func, num_runs=num_runs, num_additonal_args=num_add_args)

        # print results and compute percentage 

        print(f"\nBenchmarking with {num_add_args} additional arguments:")
        print(f"Duration without annotations:        {duration_not_annotated*1000:.4f} ms, 100.00%")
        print(f"Duration with global check never:    {duration_global_check_never*1000:.4f} ms, {duration_global_check_never/duration_not_annotated*100:5.2f}%")
        print(f"Duration with global check once:     {duration_global_check_once*1000:.4f} ms, {duration_global_check_once/duration_not_annotated*100:5.2f}%")
        print(f"Duration with shape checking:        {duration_with*1000:.4f} ms, {duration_with/duration_not_annotated*100:5.2f}%")
        print("-" * 50)