# smcount_nvml.py
from pynvml import *

NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 11  # NVML attribute enum

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)

name = nvmlDeviceGetName(h)
if isinstance(name, bytes):
    name = name.decode("utf-8", errors="replace")

cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(h)

# Generic attribute getter (present in more bindings than the dedicated function)
sm_count = nvmlDeviceGetAttribute(h, NVML_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

print("GPU:", name)
print("Compute capability:", f"{cc_major}.{cc_minor}")
print("SM count:", sm_count)

nvmlShutdown()
