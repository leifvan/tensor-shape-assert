from src.tensor_shape_assert import ShapedTensor, check_tensor_shapes
from test_utils import get_library_by_name, NAME_LIBRARY_MAP
from time import time
from tqdm import tqdm
from tabulate import tabulate


def benchmark(f, num_additonal_args, xp):

    # prepare args
    x = xp.zeros((10, 20, 30))
    y = xp.zeros((20, 2))
    add = [xp.zeros((10, 5)) for _ in range(num_additonal_args)]

    # some warmup
    for _ in range(2 ** 7):
        f(x, y, *add)

    # actual benchmark
    start_time = time()
    for i in tqdm(range(2 ** 16)):
        f(x, y, *add)

        if i % 500 == 0:
            if time() - start_time > 5:
                break
    return (time() - start_time) / (i + 1)  # type: ignore

def func(
          x: ShapedTensor["a b c"],
          y: ShapedTensor["b d"],
          *args: tuple[ShapedTensor["10 5"], ...]
) -> ShapedTensor["a d"]:
    z = x[:, :, :, None] * y[None, :, None]
    return z.sum(axis=(1, 2)) # type: ignore


if __name__ == "__main__":
    func_always = check_tensor_shapes(check_mode="always")(func)
    func_once = check_tensor_shapes(check_mode="once")(func)
    func_never = check_tensor_shapes(check_mode="never")(func)

    results = []

    def add_results(lib, num_add_args, mode, duration):

        if mode == "not_annotated":
            duration_not_annotated = duration
        else:
            duration_not_annotated = None
            for r in results:
                if r["library"] == lib and r["additional args"] == num_add_args and r["check mode"] == "not_annotated":
                    duration_not_annotated = r["duration (ms)"] / 1000
                    break

        results.append({
            "library": lib,
            "additional args": num_add_args,
            "check mode": mode,
            "duration (ms)": duration * 1000,
            "overhead (ms)": (duration - duration_not_annotated) * 1000,
            "relative (%)": duration / duration_not_annotated
        })

    for lib in NAME_LIBRARY_MAP.keys():
        try:
            xp = get_library_by_name(lib)
        except ModuleNotFoundError:
            print("skipping", lib, ", not installed")
            continue

        try:
            for num_add_args in (0, 10, 100):

                add_results(lib, num_add_args, "not_annotated", benchmark(func, num_additonal_args=num_add_args, xp=xp))
                add_results(lib, num_add_args, "never", benchmark(func_never, num_additonal_args=num_add_args, xp=xp))
                add_results(lib, num_add_args, "once", benchmark(func_once, num_additonal_args=num_add_args, xp=xp))
                add_results(lib, num_add_args, "always", benchmark(func_always, num_additonal_args=num_add_args, xp=xp))

                # if lib == "torch":
                #     import torch
                #     add_results("torch-compile", num_add_args, "not_annotated", benchmark(torch.compile(func), num_additonal_args=num_add_args, xp=xp))
                #     add_results("torch-compile", num_add_args, "never", benchmark(torch.compile(func_never), num_additonal_args=num_add_args, xp=xp))
                #     add_results("torch-compile", num_add_args, "once", benchmark(torch.compile(func_once), num_additonal_args=num_add_args, xp=xp))
                #     add_results("torch-compile", num_add_args, "always", benchmark(torch.compile(func_always), num_additonal_args=num_add_args, xp=xp))

                print(tabulate(results, tablefmt="github", floatfmt=(None, None, None, ".5f", ".4e", ".2%"))) # type: ignore
        except Exception as e:
            print("Error benchmarking", lib, ":", e)

    # write results to csv file

    with open("benchmark_results.csv", "w") as f:
        f.write("library,additional args,check mode,duration (ms),relative (%),overhead (ms)\n")
        for r in results:
            f.write(f"{r['library']},{r['additional args']},{r['check mode']},{r['duration (ms)']:.5f},{r['relative (%)']:.2%},{r['overhead (ms)']:.10f}\n")
