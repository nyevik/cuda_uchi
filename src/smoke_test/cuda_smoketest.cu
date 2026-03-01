#include <cstdio>
#include <cuda_runtime.h>

__global__ void k(int *out) {
  out[0] = 1234;
}

int main() {
  int *d = nullptr;
  int h = 0;
  cudaError_t e;

  e = cudaMalloc(&d, sizeof(int));
  if (e != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(e)); return 1; }

  k<<<1,1>>>(d);
  e = cudaGetLastError();
  if (e != cudaSuccess) { printf("kernel launch failed: %s\n", cudaGetErrorString(e)); return 1; }

  e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) { printf("cudaMemcpy failed: %s\n", cudaGetErrorString(e)); return 1; }

  cudaFree(d);
  printf("OK: %d\n", h);
  return 0;
}
