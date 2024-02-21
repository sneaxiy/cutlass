#include <type_traits>
#include "fast_matmul.h"
#include "cutlass/gemm/device/gemm.h"

template <typename T, bool TransposeX, bool TransposeY>
struct CutlassGEMMTrait {
  using ScalarType = T; 
  using AccumType = T; 
  using Type = T; 
  using Op = ::cutlass::arch::OpClassSimt;
  using Arch = ::cutlass::arch::Sm80;
};

template <bool TransposeX, bool TransposeY>
struct CutlassGEMMTrait<cutlass::half_t, TransposeX, TransposeY> {
  using ScalarType = float;
  using AccumType = float;
  using Type = ::cutlass::half_t;
  using Op = ::cutlass::arch::OpClassTensorOp;
  using Arch = ::cutlass::arch::Sm80;
};

template <bool TransposeX, bool TransposeY>
struct CutlassGEMMTrait<cutlass::bfloat16_t, TransposeX, TransposeY> {
  using ScalarType = float;
  using AccumType = float;
  using Type = ::cutlass::bfloat16_t;
  using Op = ::cutlass::arch::OpClassTensorOp;
  using Arch = ::cutlass::arch::Sm80;
};

template <typename T, bool TransposeX, bool TransposeY>
struct CutlassGEMM {
  using CutlassDType =
      typename CutlassGEMMTrait<T, TransposeX, TransposeY>::Type;
  using GEMM = ::cutlass::gemm::device::Gemm<
      CutlassDType,
      typename std::conditional<TransposeX,
                                ::cutlass::layout::ColumnMajor,
                                ::cutlass::layout::RowMajor>::type,
      CutlassDType,
      typename std::conditional<TransposeY,
                                ::cutlass::layout::ColumnMajor,
                                ::cutlass::layout::RowMajor>::type,
      CutlassDType,
      ::cutlass::layout::RowMajor,
      typename CutlassGEMMTrait<T, TransposeX, TransposeY>::AccumType,
      typename CutlassGEMMTrait<T, TransposeX, TransposeY>::Op,
      typename CutlassGEMMTrait<T, TransposeX, TransposeY>::Arch>;
};

template <typename T, typename ScalarT, bool TransposeA, bool TransposeB>
bool FastMatmulImpl(const T *A, 
                    const T *B,
                    const T *C,
                    T *D,
                    int M,
                    int N,
                    int K,
                    ScalarT alpha,
                    ScalarT beta,
                    void *workspace,
                    size_t *workspace_size,
                    cudaStream_t stream) {
  auto lda = TransposeA ? M : K;
  auto ldb = TransposeB ? K : N;
  auto ldc = N;
  
  using GEMM = typename CutlassGEMM<T, TransposeA, TransposeB>::GEMM;
  GEMM gemm_op;
  typename GEMM::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {D, ldc}, {alpha, beta}); 
  if (workspace_size == nullptr) {
    auto status = gemm_op(args, workspace, stream); 
    return status == cutlass::Status::kSuccess;
  } else {
    *workspace_size = GEMM::get_workspace_size(args); 
    return true;
  }
}


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
                cudaStream_t stream) {
#define CALL_FAST_MATMUL_BASE_IMPL(__dtype, __cpp_dtype, __transpose_a, __transpose_b) \
  do { \
    if (dtype == __dtype && transpose_a == __transpose_a && transpose_b == __transpose_b) { \
      using __Type = typename CutlassGEMMTrait<__cpp_dtype, __transpose_a, __transpose_b>::Type; \
      using __ScalarType = typename CutlassGEMMTrait<__cpp_dtype, __transpose_a, __transpose_b>::ScalarType; \
      return FastMatmulImpl<__cpp_dtype, __ScalarType, __transpose_a, __transpose_b>(static_cast<const __cpp_dtype*>(A), \
          static_cast<const __cpp_dtype*>(B), static_cast<const __cpp_dtype*>(C), \
          static_cast<__cpp_dtype*>(D), M, N, K, static_cast<__ScalarType>(alpha), static_cast<__ScalarType>(beta), \
          workspace, workspace_size, stream); \
    } \
  } while (0)


#define CALL_FAST_MATMUL_IMPL(__dtype, __cpp_dtype) \
  CALL_FAST_MATMUL_BASE_IMPL(__dtype, __cpp_dtype, false, false); \
  CALL_FAST_MATMUL_BASE_IMPL(__dtype, __cpp_dtype, false, true); \
  CALL_FAST_MATMUL_BASE_IMPL(__dtype, __cpp_dtype, true, false); \
  CALL_FAST_MATMUL_BASE_IMPL(__dtype, __cpp_dtype, true, true)

  CALL_FAST_MATMUL_IMPL(FastMatmulDataType::BFLOAT16, cutlass::bfloat16_t);
  CALL_FAST_MATMUL_IMPL(FastMatmulDataType::FLOAT16, cutlass::half_t); 
  CALL_FAST_MATMUL_IMPL(FastMatmulDataType::FLOAT32, float);
  CALL_FAST_MATMUL_IMPL(FastMatmulDataType::FLOAT64, double);
  return false;
}
