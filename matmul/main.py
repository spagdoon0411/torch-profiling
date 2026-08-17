import einops as ein
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.benchmark
import torch.utils.flop_counter
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


class ByteCounterMode(TorchDispatchMode):
    """Sums bytes of every tensor touched (args + output) by each op.

    Kind of serves as a memory-traffic analog of FlopCounterMode: an
    idealized, no-reuse/no-fusion upper bound derived from actual runtime
    shapes, not a hardware measurement of DRAM bytes moved.
    """

    def __init__(self):
        super().__init__()
        self.total_bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        # Only tally ops FlopCounterMode also tracks (e.g. mm, bmm, addmm), so
        # the byte count and flop count describe the same set of operations —
        # otherwise bytes would include reshape/view/copy ops flops ignores.
        if func._overloadpacket in torch.utils.flop_counter.flop_registry:

            def count(t):
                if isinstance(t, torch.Tensor):
                    self.total_bytes += t.numel() * t.element_size()

            tree_map(count, args)
            tree_map(count, kwargs)
            tree_map(count, out)
        return out


def contraction(a, b):
    return ein.einsum(a, b, "i j, j k -> i k")


def count_flops_and_bytes(a, b) -> tuple[float, float]:
    with (
        torch.utils.flop_counter.FlopCounterMode(display=False) as fc,
        ByteCounterMode() as bc,
    ):
        _ = contraction(a, b)

    return fc.get_total_flops(), bc.total_bytes


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_context(shape, dtype, device, min_run_time):
    print(f"shape={shape} dtype={dtype} device={device} min_run_time={min_run_time}s")


def run_benchmark(
    shape, dtype
) -> tuple[torch.utils.benchmark.Measurement, float, float]:
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

    # Yes, you could compute the theoretical values here because it's a simple
    # operation and you know the input tensors. It was nice to try PyTorch's
    # empirical tools, however.
    flops, bytes_ = count_flops_and_bytes(a, b)

    timer = torch.utils.benchmark.Timer(
        stmt="contraction(a, b)",
        globals={
            "a": a,
            "b": b,
            "contraction": contraction,
        },
    )

    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    return measurement, flops, bytes_


def main():
    shapes = [2**n for n in range(13)]
    times = []
    flops = []
    bytes_ = []

    for shape in shapes:
        res_meas, res_flops, res_bytes = run_benchmark((shape, shape), torch.float16)
        times.append(res_meas.median)
        flops.append(res_flops)
        bytes_.append(res_bytes)

    intensity = np.array(flops) / np.array(bytes_)  # FLOPs per byte moved

    plt.figure()
    plt.plot(shapes, times)
    plt.xlabel("shape (n, n square matmul)")
    plt.ylabel("median time (s)")
    plt.savefig(
        "matmul/times.png"
    )  # Produces an exponential curve \propto n^3 (as expected)

    plt.figure()
    plt.plot(shapes, intensity)
    plt.xlabel("shape (n, n square matmul)")
    plt.ylabel("arithmetic intensity (FLOPs / byte)")
    plt.xscale("log", base=2)
    plt.savefig("matmul/arithmetic_intensity.png")


if __name__ == "__main__":
    main()
