/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>

#include <iostream>
#include <string>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
// #include <cuda_runtime_api.h>
#include <c10/util/Exception.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_size.h>
#include <cutlass/numeric_types.h>
// #include <tensorrt_llm/common/assert.h>
#include "flashinfer/trtllm/common/cudaUtils.h"

namespace moe::dev {

#define CHECK_CUDA(cmd)                                                                      \
  do {                                                                                       \
    cudaError_t e = cmd;                                                                     \
    if (e != cudaSuccess) {                                                                  \
      std::cout << "CUDA error in " << __FILE__ << ":" << __LINE__ << " executing '" << #cmd \
                << "': " << cudaGetErrorString(e);                                           \
    }                                                                                        \
    TORCH_CHECK(e == cudaSuccess, "Got CUDA error. See above for details.");                 \
  } while (0)
////////////////////////////////////////////////////////////////////////////////////////////////////

#define LAUNCH_ESC(...) __VA_ARGS__

#define LAUNCH_PDL(data, coopLaunch, types, kernel, numBlocks, numThreads, smemSize, stream)    \
  cudaLaunchConfig_t config{};                                                                  \
  config.gridDim = numBlocks;                                                                   \
  config.blockDim = numThreads;                                                                 \
  config.dynamicSmemBytes = smemSize;                                                           \
  config.stream = (cudaStream_t)stream;                                                         \
                                                                                                \
  cudaLaunchAttribute attributes[2] = {};                                                       \
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                        \
  attributes[0].val.programmaticStreamSerializationAllowed = int(data.mUsePdl);                 \
  attributes[1].id = cudaLaunchAttributeCooperative;                                            \
  attributes[1].val.cooperative = int(coopLaunch);                                              \
  config.attrs = attributes;                                                                    \
  config.numAttrs = 2;                                                                          \
  if (data.mUsePdl) {                                                                           \
    auto params = KernelParams<types, true>::setKernelParams(data);                             \
    auto kernelTyped = kernel<KernelParams<types, true>>;                                       \
    if (smemSize > 48 * 1024) {                                                                 \
      CHECK_CUDA(cudaFuncSetAttribute(kernelTyped, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smemSize));                                               \
    }                                                                                           \
    CHECK_CUDA(cudaLaunchKernelEx(&config, kernelTyped, params));                               \
  } else {                                                                                      \
    auto params = KernelParams<types, false>::setKernelParams(data);                            \
    auto kernelTyped = kernel<KernelParams<types, false>>;                                      \
    if (smemSize > 48 * 1024) {                                                                 \
      CHECK_CUDA(cudaFuncSetAttribute(kernelTyped, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smemSize));                                               \
    }                                                                                           \
    CHECK_CUDA(cudaLaunchKernelEx(&config, kernelTyped, params));                               \
  }

#define LAUNCH(data, kernel, numBlocks, numThreads, smemSize, stream)                              \
  if (data.mDtypeElt == tg::Dtype::Fp16) {                                                         \
    LAUNCH_PDL(data, false, cutlass::half_t, kernel, numBlocks, numThreads, smemSize, stream);     \
  } else if (data.mDtypeElt == tg::Dtype::E4m3) {                                                  \
    LAUNCH_PDL(data, false, cutlass::float_e4m3_t, kernel, numBlocks, numThreads, smemSize,        \
               stream);                                                                            \
  } else if (data.mDtypeElt == tg::Dtype::Bfloat16) {                                              \
    LAUNCH_PDL(data, false, cutlass::bfloat16_t, kernel, numBlocks, numThreads, smemSize, stream); \
  } else {                                                                                         \
    TORCH_WARN("Unsupported dtypeElt");                                                            \
  }

#define LAUNCH_EXPW(data, kernel, numBlocks, numThreads, smemSize, stream)                         \
  if (data.mDtypeElt == tg::Dtype::Fp16 && data.mDtypeExpW == tg::Dtype::Fp32) {                   \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::half_t, float), kernel, numBlocks, numThreads,     \
               smemSize, stream);                                                                  \
  } else if (data.mDtypeElt == tg::Dtype::E4m3 && data.mDtypeExpW == tg::Dtype::Fp32) {            \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::float_e4m3_t, float), kernel, numBlocks,           \
               numThreads, smemSize, stream);                                                      \
  } else if (data.mDtypeElt == tg::Dtype::Bfloat16 && data.mDtypeExpW == tg::Dtype::Fp32) {        \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::bfloat16_t, float), kernel, numBlocks, numThreads, \
               smemSize, stream);                                                                  \
  } else if (data.mDtypeElt == tg::Dtype::Fp16 && data.mDtypeExpW == tg::Dtype::Bfloat16) {        \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::half_t, cutlass::bfloat16_t), kernel, numBlocks,   \
               numThreads, smemSize, stream);                                                      \
  } else if (data.mDtypeElt == tg::Dtype::E4m3 && data.mDtypeExpW == tg::Dtype::Bfloat16) {        \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::float_e4m3_t, cutlass::bfloat16_t), kernel,        \
               numBlocks, numThreads, smemSize, stream);                                           \
  } else if (data.mDtypeElt == tg::Dtype::Bfloat16 && data.mDtypeExpW == tg::Dtype::Bfloat16) {    \
    LAUNCH_PDL(data, false, LAUNCH_ESC(cutlass::bfloat16_t, cutlass::bfloat16_t), kernel,          \
               numBlocks, numThreads, smemSize, stream);                                           \
  } else {                                                                                         \
    TORCH_WARN("Unsupported pair");                                                                \
  }

#define LAUNCH_ROUTING(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream)     \
  if (data.mDtypeExpW == tg::Dtype::Fp32) {                                                   \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(float, float), kernel, numBlocks, numThreads,     \
               smemSize, stream);                                                             \
  } else if (data.mDtypeExpW == tg::Dtype::Bfloat16) {                                        \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16), kernel, numBlocks, \
               numThreads, smemSize, stream);                                                 \
  } else {                                                                                    \
    TORCH_WARN("Unsupported dtypeExpW");                                                      \
  }

#define LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, \
                                       stream, extraFlag, forceFloatInput)                        \
  if (data.mDtypeExpW == tg::Dtype::Fp32 && extraFlag) {                                          \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(float, float, true), kernel, numBlocks, numThreads,   \
               smemSize, stream);                                                                 \
  } else if (data.mDtypeExpW == tg::Dtype::Fp32) {                                                \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(float, float, false), kernel, numBlocks, numThreads,  \
               smemSize, stream);                                                                 \
  } else if (data.mDtypeExpW == tg::Dtype::Bfloat16 && extraFlag && forceFloatInput) {            \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(float, __nv_bfloat16, true), kernel, numBlocks,       \
               numThreads, smemSize, stream);                                                     \
  } else if (data.mDtypeExpW == tg::Dtype::Bfloat16 && extraFlag) {                               \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, true), kernel,          \
               numBlocks, numThreads, smemSize, stream);                                          \
  } else if (data.mDtypeExpW == tg::Dtype::Bfloat16 && forceFloatInput) {                         \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(float, __nv_bfloat16, false), kernel, numBlocks,      \
               numThreads, smemSize, stream);                                                     \
  } else if (data.mDtypeExpW == tg::Dtype::Bfloat16) {                                            \
    LAUNCH_PDL(data, coopLaunch, LAUNCH_ESC(__nv_bfloat16, __nv_bfloat16, false), kernel,         \
               numBlocks, numThreads, smemSize, stream);                                          \
  } else {                                                                                        \
    TORCH_WARN("Unsupported dtypeExpW");                                                          \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data {
  tg::Dtype mDtypeElt{tg::Dtype::Fp16};
  bool mUsePdl{false};
  bool mUseDeepSeekFp8{false};

  void* inPtr;
  void* outPtr;
  float* inDqSfsPtr = nullptr;
  float* outDqSfsPtr = nullptr;

  int32_t innerDim;
  int32_t numTokens;
  int32_t topK;
  int32_t* expandedIdxToPermutedIdx;

  int32_t const* totalNumPaddedTokens;
};

template <typename Type_, bool UsePdl_>
struct KernelParams {
  using Type = Type_;
  static constexpr bool UsePdl = UsePdl_;

  Type const* inPtr;
  Type* outPtr;

  float* inDqSfsPtr = nullptr;
  float* outDqSfsPtr = nullptr;

  int32_t innerDim;
  int32_t numTokens;
  int32_t topK;
  int32_t* expandedIdxToPermutedIdx;

  int32_t const* totalNumPaddedTokens;

  static KernelParams setKernelParams(Data const& data) {
    KernelParams params;

    params.inPtr = (Type*)data.inPtr;
    params.outPtr = (Type*)data.outPtr;
    params.inDqSfsPtr = data.inDqSfsPtr;
    params.outDqSfsPtr = data.outDqSfsPtr;

    params.expandedIdxToPermutedIdx = data.expandedIdxToPermutedIdx;

    params.innerDim = data.innerDim;
    params.numTokens = data.numTokens;
    params.topK = data.topK;
    params.totalNumPaddedTokens = data.totalNumPaddedTokens;

    return params;
  }
};

void run(Data const& data, void* stream);

}  // namespace activation

namespace convertsf {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data {
  bool mUsePdl{false};

  void* inSfPtr = nullptr;
  void* outSfPtr = nullptr;
  int32_t hiddenDimSf;
  int32_t numTokens;
  tg::SfLayout sfLayoutSrc;
  tg::SfLayout sfLayoutDst;
};

template <typename Type_, bool UsePdl_>
struct KernelParams {
  using Type = Type_;
  static constexpr bool UsePdl = UsePdl_;

  void const* inSfPtr = nullptr;
  void* outSfPtr = nullptr;
  int32_t hiddenDimSf;
  int32_t numTokens;
  tg::SfLayout sfLayoutSrc;
  tg::SfLayout sfLayoutDst;

  static KernelParams setKernelParams(Data const& data) {
    KernelParams params;

    params.inSfPtr = data.inSfPtr;
    params.outSfPtr = data.outSfPtr;
    params.hiddenDimSf = data.hiddenDimSf;
    params.numTokens = data.numTokens;
    params.sfLayoutSrc = data.sfLayoutSrc;
    params.sfLayoutDst = data.sfLayoutDst;

    return params;
  }
};

void run(Data const& data, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace convertsf

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace permute {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data {
  tg::Dtype mDtypeElt{tg::Dtype::Fp16};
  bool mUsePdl{false};
  bool mUseDeepSeekFp8{false};

  void* inPtr;
  void* outPtr;
  float* inDqSfsPtr = nullptr;
  float* outDqSfsPtr = nullptr;
  int32_t* expandedIdxToPermutedIdx;
  int32_t hiddenDim;
  int32_t numTokens;
  int32_t topK;
  int32_t const* totalNumPaddedTokens;
};

template <typename Type_, bool UsePdl_>
struct KernelParams {
  using Type = Type_;
  static constexpr bool UsePdl = UsePdl_;

  Type const* inPtr;
  Type* outPtr;
  float const* inDqSfsPtr;
  float* outDqSfsPtr;
  int32_t* expandedIdxToPermutedIdx;
  int32_t hiddenDim;
  int32_t numTokens;
  int32_t topK;
  int32_t const* totalNumPaddedTokens;
  bool useDeepSeekFp8;

  static KernelParams setKernelParams(Data const& data) {
    KernelParams params;

    params.inPtr = (Type*)data.inPtr;
    params.outPtr = (Type*)data.outPtr;
    params.inDqSfsPtr = data.inDqSfsPtr;
    params.outDqSfsPtr = data.outDqSfsPtr;
    params.expandedIdxToPermutedIdx = data.expandedIdxToPermutedIdx;
    params.hiddenDim = data.hiddenDim;
    params.numTokens = data.numTokens;
    params.topK = data.topK;
    params.totalNumPaddedTokens = data.totalNumPaddedTokens;
    params.useDeepSeekFp8 = data.mUseDeepSeekFp8;

    return params;
  }
};

void run(Data const& data, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Data {
  tg::Dtype mDtypeElt{tg::Dtype::Fp16};
  tg::Dtype mDtypeExpW{tg::Dtype::Bfloat16};
  bool mUsePdl{false};
  bool mUseDeepSeekFp8{false};

  void* inPtr;
  void* outPtr;
  float* inDqSfsPtr = nullptr;
  float* outDqSfsPtr = nullptr;

  void* expertWeightsPtr;
  int32_t* expandedIdxToPermutedIdx;

  int32_t numTokens;
  int32_t numExperts;
  int32_t topK;
  int32_t hiddenDim;
  int32_t const* totalNumPaddedTokens;
};

template <typename Type_, typename TypeExpW_, bool UsePdl_>
struct KernelParams {
  using Type = Type_;
  using TypeExpW = TypeExpW_;
  static constexpr bool UsePdl = UsePdl_;

  Type const* inPtr;
  TypeExpW const* expertWeightsPtr;
  Type* outPtr;

  float* inDqSfsPtr = nullptr;
  float* outDqSfsPtr = nullptr;

  int32_t* expandedIdxToPermutedIdx;

  int32_t hiddenDim;
  int32_t numTokens;
  int32_t numExperts;
  int32_t topK;
  int32_t const* totalNumPaddedTokens;

  static KernelParams setKernelParams(Data const& data) {
    KernelParams params;

    params.inPtr = (Type*)data.inPtr;
    params.expertWeightsPtr = (TypeExpW*)data.expertWeightsPtr;
    params.outPtr = (Type*)data.outPtr;
    params.inDqSfsPtr = data.inDqSfsPtr;
    params.outDqSfsPtr = data.outDqSfsPtr;

    params.expandedIdxToPermutedIdx = data.expandedIdxToPermutedIdx;

    params.hiddenDim = data.hiddenDim;
    params.numTokens = data.numTokens;
    params.numExperts = data.numExperts;
    params.topK = data.topK;
    params.totalNumPaddedTokens = data.totalNumPaddedTokens;

    return params;
  }
};

void run(Data const& data, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace moe::dev
