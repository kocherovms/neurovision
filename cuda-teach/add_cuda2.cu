#include <iostream>
#include <math.h>

__global__
void add(const int theN, float * theX, float * const theY) {
  const int index = threadIdx.x;
  const int stride = blockDim.x;
  
  for(int i = index; i < theN; i += stride) {
    theY[i] = theX[i] + theY[i];
  }
}

int main() {
  const int N = 1 << 20;
  float * x, * y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for(int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  add<<<1, 1024>>>(N, x, y);
  cudaDeviceSynchronize();

  float maxError = 0.0f;

  for(int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }

  std::cout << "maxError=" << maxError << std::endl;

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
