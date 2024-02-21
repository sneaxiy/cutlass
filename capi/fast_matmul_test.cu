#include "fast_matmul.h"
#include <iostream>
#include <chrono>


#define CUDA_CHECK(__status)                                              \
  do {                                                                    \
    cudaError_t __error = __status;                                       \
    if (__error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(__error) \
                << " at line: " << __LINE__ << std::endl;                 \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)


static size_t SizeofDType(FastMatmulDataType dtype) {
  switch (dtype) {
    case FLOAT64: return 8;
    case FLOAT32: return 4;
    case FLOAT16: return 2;
    case BFLOAT16: return 2;
    default:
     throw std::runtime_error("Unsupported data type");
  }
} 

template <FastMatmulDataType DType> 
void FastMatmulTestImpl(int M, int N, int K, bool transpose_a, bool transpose_b) {
  double alpha = 1, beta = 0;
  size_t warmup_steps = 20, trial_steps = 1000;

  void *A, *B, *C, *D; 
  cudaStream_t stream;
  void *workspace = nullptr;
  size_t workspace_size = 0;

  auto sizeof_dtype = SizeofDType(DType);
  CUDA_CHECK(cudaMalloc(&A, sizeof_dtype * M * K)); 
  CUDA_CHECK(cudaMalloc(&B, sizeof_dtype * K * N));
  CUDA_CHECK(cudaMalloc(&C, sizeof_dtype * M * N));
  CUDA_CHECK(cudaMalloc(&D, sizeof_dtype * M * N));
  CUDA_CHECK(cudaStreamCreate(&stream));

  bool status = FastMatmul(A, B, C, D, M, N, K, transpose_a, transpose_b, DType, alpha, beta, workspace, &workspace_size, stream);
  if (!status) {
    throw std::runtime_error("Get workspace size error");
  }

  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  auto func = [&] {
    status = FastMatmul(A, B, C, D, M, N, K, transpose_a, transpose_b, DType, alpha, beta, workspace, nullptr, stream); 
    if (!status) {
      throw std::runtime_error("Call FastMatmul error");
    }
  };

  for (size_t i = 0; i < warmup_steps; ++i) {
    func();
  } 
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < trial_steps; ++i) {
    func();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::high_resolution_clock::now(); 
  auto cost = 1.0 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / trial_steps);
  auto flops = 2.0 * M * N * K / (cost * 312e3); 
  std::cout << "FLOPS = " << flops << " , time = " << cost << std::endl;

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(D));
  CUDA_CHECK(cudaFree(workspace));
}

template <FastMatmulDataType DType>
void FastMatmulTest(int M, int N, int K) {
  FastMatmulTestImpl<DType>(M, N, K, false, false);
  FastMatmulTestImpl<DType>(M, K, N, false, true); 
  FastMatmulTestImpl<DType>(K, M, N, true, false);   
} 

int main() {
  FastMatmulTest<FastMatmulDataType::BFLOAT16>(8192, 12800, 6400);
  return 0;
}
