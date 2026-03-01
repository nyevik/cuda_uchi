import cupy as cp

dev = cp.cuda.Device(0)
attrs = dev.attributes
print("SM count:", attrs["MultiProcessorCount"])
