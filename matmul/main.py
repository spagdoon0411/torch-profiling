import einops as ein
import matplotlib.pyplot as plt
import torch
import torch.utils.benchmark


def contraction(a, b):
    _ = ein.einsum(a, b, "i j, j k -> i k")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_context(shape, dtype, device, min_run_time):
    print(f"shape={shape} dtype={dtype} device={device} min_run_time={min_run_time}s")


def run_benchmark(shape, dtype) -> torch.utils.benchmark.Measurement:
    min = 0
    max = 16
    min_run_time = 5.0  # Seconds
    device = choose_device()

    print_context(shape, dtype, device, min_run_time)

    a = min + (max - min) * torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )
    b = min + (max - min) * torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    timer = torch.utils.benchmark.Timer(
        stmt="contraction(a, b)",
        globals={"a": a, "b": b, "contraction": contraction},
    )

    return timer.blocked_autorange(min_run_time=min_run_time)


def main():
    shapes = [2**n for n in range(13)]
    times = []

    for shape in shapes:
        result = run_benchmark((shape, shape), torch.float16)
        times.append(result.median)

    plt.plot(shapes, times)
    plt.savefig(
        "matmul/times.png"
    )  # Produces an exponential curve \propto 2^n (as expected)


if __name__ == "__main__":
    main()
