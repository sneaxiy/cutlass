#pragma once

#include "cuda_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif

enum FastMatmulDataType {
  FLOAT64 = 0,
  FLOAT32 = 1,
  FLOAT16 = 2,
  BFLOAT16 = 3 
}; 

bool FastMatmul(const void *A,
                const void *B,
                const void *C,
                void *D,
                int M,
                int N,
                int K,
                bool transpose_a,
                bool transpose_b,
                FastMatmulDataType dtype,
                double alpha,
                double beta,
                void *workspace,
                size_t *workspace_size,
                cudaStream_t stream);
                 
#ifdef __cplusplus
}
#endif
