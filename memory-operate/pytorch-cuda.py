from torch.profiler import profile, ProfilerActivity
import torch
import gc
import matplotlib.pyplot as plt

from torch.utils.benchmark import Timer
from caesar.logger.logger import setup_logger

logger = setup_logger(__name__)


def timer(cmd):
    median = (
        Timer(cmd, globals=globals())
        .adaptive_autorange(min_run_time=1.0, max_run_time=20.0)
        .median*1000
    )
    logger.info(f"{cmd}: {median:4.4f} ms")
    return median


pageable_tensor = torch.randn(1_000_000)

pinned_tensor = torch.randn(1_000_000, pin_memory=True)

pageable_to_device = timer("pageable_tensor.to('cuda:0')")
pinned_to_device = timer("pinned_tensor.to('cuda:0')")
pin_mem = timer("pageable_tensor.pin_memory()")
pin_mem_to_device = timer("pageable_tensor.pin_memory().to('cuda:0')")


r1 = pinned_to_device/pageable_to_device
r2 = pin_mem_to_device/pageable_to_device

fig, ax = plt.subplots()

xlabels = [0, 1, 2]
bar_labels = [
    "pageable_tensor.to(device) (1x)",
    f"pinned_tensor.to(device) ({r1:4.2f}x)",
    f"pageable_tensor.pin_memory().to(device) ({r2:4.2f}x)"
    f"\npin_memory()={100*pin_mem/pin_mem_to_device:.2f}% of runtime.",
]

values = [pageable_to_device, pinned_to_device, pin_mem_to_device]
colors = ["tab:blue", "tab:red", "tab:orange"]
ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime (pin-memory)")
ax.set_xticks([])
ax.legend()

plt.show()

# Clear tensors
del pageable_tensor, pinned_tensor
_ = gc.collect()


# A simple loop that copies all tensors to cuda
def copy_to_device(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0"))
    return result


# A loop that copies all tensors to cuda asynchronously
def copy_to_device_nonblocking(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.to("cuda:0", non_blocking=True))
    # We need to synchronize
    torch.cuda.synchronize()
    return result


# Create a list of tensors
tensors = [torch.randn(1000) for _ in range(1000)]
to_device = timer("copy_to_device(*tensors)")
to_device_nonblocking = timer("copy_to_device_nonblocking(*tensors)")

# Ratio
r1 = to_device_nonblocking / to_device

# Plot the results
fig, ax = plt.subplots()

xlabels = [0, 1]
bar_labels = [f"to(device) (1x)",
              f"to(device, non_blocking=True) ({r1:4.2f}x)"]
colors = ["tab:blue", "tab:red"]
values = [to_device, to_device_nonblocking]

ax.bar(xlabels, values, label=bar_labels, color=colors)

ax.set_ylabel("Runtime (ms)")
ax.set_title("Device casting runtime (non-blocking)")
ax.set_xticks([])
ax.legend()

plt.show()


def profile_mem(cmd):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        exec(cmd)
    print(cmd)
    print(prof.key_averages().table(row_limit=10))


logger.info("Call to `to(device)`")
profile_mem("copy_to_device(*tensors)")
logger.info("Call to `to(device, non_blocking=True)`")
profile_mem("copy_to_device_nonblocking(*tensors)")
