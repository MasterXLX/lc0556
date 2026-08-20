// Attempt 2: simple one-thread-per-output reference implementation.
// This file is intentionally isolated from common_kernels.cu so it can be
// compared against the optimized direct-port candidate without displacing
// existing uncertainty-weighting code.

#include "cuda_common.h"

namespace lczero {
namespace cudnn_backend {

__device__ __forceinline__ int rpeIndex4(int i, int j, int k, int l,
                                         int I, int J, int K, int L) {
  if (i >= I || j >= J || k >= K || l >= L) return -1;
  return (((i * J + j) * K + k) * L) + l;
}

template <typename T>
__global__ void rpeReferenceQKV(const T* rpeInput, const T* rpeWeights,
                                const T* skipAdd, T* output, int B, int H,
                                int Q, int K, int D, float outScale,
                                int rpetype) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (rpetype == 0 || rpetype == 1) {
    int total = B * H * Q * K;
    if (tid >= total) return;
    int k = tid % K;
    int q = (tid / K) % Q;
    int h = (tid / (Q * K)) % H;
    int b = tid / (H * Q * K);
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
      int in = rpetype == 0 ? rpeIndex4(b,q,h,d,B,Q,H,D)
                            : rpeIndex4(b,k,h,d,B,K,H,D);
      int w  = rpetype == 0 ? rpeIndex4(h,q,k,d,H,Q,K,D)
                            : rpeIndex4(h,k,q,d,H,K,Q,D);
      sum += (float)rpeInput[in] * (float)rpeWeights[w];
    }
    output[tid] = (T)(((float)skipAdd[tid] + sum) * outScale);
    return;
  }

  int total = B * Q * H * D;
  if (tid >= total) return;
  int d = tid % D;
  int h = (tid / D) % H;
  int q = (tid / (H * D)) % Q;
  int b = tid / (Q * H * D);
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    int in = rpeIndex4(b,h,q,k,B,H,Q,K);
    int w  = rpeIndex4(h,q,d,k,H,Q,D,K);
    sum += (float)rpeInput[in] * (float)rpeWeights[w];
  }
  output[tid] = (T)(((float)skipAdd[tid] + sum) * outScale);
}

template <typename T>
void multiplyRPEReference(const T* rpeInput, const T* rpeWeights,
                          const T* attnInput, T* output, int B, int H, int Q,
                          int K, int D, float outScale, int rpetype,
                          cudaStream_t stream) {
  if (rpetype < 0 || rpetype > 2)
    throw Exception("unsupported rpetype in multiplyRPEReference");
  int total = rpetype == 2 ? B * Q * H * D : B * H * Q * K;
  int blocks = DivUp(total, 256);
  rpeReferenceQKV<<<blocks,256,0,stream>>>(rpeInput,rpeWeights,attnInput,
                                           output,B,H,Q,K,D,outScale,rpetype);
  ReportCUDAErrors(cudaGetLastError());
}

}  // namespace cudnn_backend
}  // namespace lczero
