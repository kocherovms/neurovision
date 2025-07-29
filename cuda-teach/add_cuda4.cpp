#include <iostream>
#include <math.h>

__global__
void add(const int theN, float * theX, float * const theY) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  
  for(int i = index; i < theN; i += stride) {
    theY[i] = theX[i] + theY[i];
  }
}

int main() {
  //const int N = 1 << 20;
  const int N = 10_000;
  float * x, * y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for(int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }


  cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
  cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

  const int blockSize = 1;
  int numBlocks = N;
  std::cout << "numBlocks=" << numBlocks << std::endl;
  add<<<numBlocks, blockSize>>>(N, x, y);
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
