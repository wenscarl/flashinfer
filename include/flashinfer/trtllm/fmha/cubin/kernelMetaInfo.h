/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <flashinfer/trtllm/fmha/kernelParams.h>
// Helper to print Data_type
inline const char* dataTypeToString(Data_type dt) {
  switch (dt) {
    case DATA_TYPE_FP16:
      return "FP16";
    case DATA_TYPE_BF16:
      return "BF16";
    case DATA_TYPE_FP32:
      return "FP32";
    case DATA_TYPE_E4M3:
      return "E4M3";
    default:
      return "UNKNOWN";
  }
}

static const struct TllmGenFmhaKernelMetaInfo {
  Data_type mDataTypeQ;
  Data_type mDataTypeKv;
  Data_type mDataTypeO;
  int mTileSizeQ;
  int mTileSizeKv;
  int mStepQ;
  int mStepKv;
  int mHeadDimPerCtaV;
  int mHeadDimQk;
  int mHeadDimV;
  int mSM;
  const char* mFuncName;
  int mSharedMemBytes;
  int mThreadsPerCTA;
  int mQkvLayout;
  int mNumTokensPerPage;
  int mMaskType;
  int mKernelType;
  int mMaxNumHeadsQPerKvInCta;
  int mTileScheduler;
  bool mGroupsHeadsQ;
  bool mMultiCtasKvMode;
  bool mReuseSmemKForV;
  bool m2CtaMma;
  const char* sha256;

  void print() const {
    std::cout << "TllmGenFmhaKernelMetaInfo {\n";
    std::cout << "  mDataTypeQ: " << dataTypeToString(mDataTypeQ) << "\n";
    std::cout << "  mDataTypeKv: " << dataTypeToString(mDataTypeKv) << "\n";
    std::cout << "  mDataTypeO: " << dataTypeToString(mDataTypeO) << "\n";
    std::cout << "  mTileSizeQ: " << mTileSizeQ << "\n";
    std::cout << "  mTileSizeKv: " << mTileSizeKv << "\n";
    std::cout << "  mStepQ: " << mStepQ << "\n";
    std::cout << "  mStepKv: " << mStepKv << "\n";
    std::cout << "  mHeadDimPerCtaV: " << mHeadDimPerCtaV << "\n";
    std::cout << "  mHeadDimQk: " << mHeadDimQk << "\n";
    std::cout << "  mHeadDimV: " << mHeadDimV << "\n";
    std::cout << "  mSM: " << mSM << "\n";
    std::cout << "  mFuncName: " << (mFuncName ? mFuncName : "null") << "\n";
    std::cout << "  mSharedMemBytes: " << mSharedMemBytes << "\n";
    std::cout << "  mThreadsPerCTA: " << mThreadsPerCTA << "\n";
    std::cout << "  mQkvLayout: " << mQkvLayout << "\n";
    std::cout << "  mNumTokensPerPage: " << mNumTokensPerPage << "\n";
    std::cout << "  mMaskType: " << mMaskType << "\n";
    std::cout << "  mKernelType: " << mKernelType << "\n";
    std::cout << "  mMaxNumHeadsQPerKvInCta: " << mMaxNumHeadsQPerKvInCta << "\n";
    std::cout << "  mTileScheduler: " << mTileScheduler << "\n";
    std::cout << "  mGroupsHeadsQ: " << std::boolalpha << mGroupsHeadsQ << "\n";
    std::cout << "  mMultiCtasKvMode: " << std::boolalpha << mMultiCtasKvMode << "\n";
    std::cout << "  mReuseSmemKForV: " << std::boolalpha << mReuseSmemKForV << "\n";
    std::cout << "  m2CtaMma: " << std::boolalpha << m2CtaMma << "\n";
    std::cout << "  sha256: " << (sha256 ? sha256 : "null") << "\n";
    std::cout << "}\n";
  }
} sTllmGenFmhaKernelMetaInfos[] = {
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ8Kv128StaticSwapsAbForGen", 75408, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "21af976bb7b61452ee2f490d974f8420d8df8bec80d66f8fce0121aa802763bb"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64VarSeqQ8Kv128StaticSwapsAbForGen", 75408, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "652abd141c50018da218ca7749d91339a77b4585705abcc1ec4a8d9d35923652"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ8Kv128StaticSwapsAbForGen", 141968, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "6c8ccca92e42d595eea104718de6bbe2be2651657083d027217ab1282d98c0f7"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16VarSeqQ8Kv128StaticSwapsAbForGen", 75408, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "5285c83a0da7b2046103d2999d768a8c3e20fed69040243298f2a13103031f66"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64VarSeqQ8Kv128StaticSwapsAbForGen", 141968, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "42da378781aaa53febab0204fae44d3e0f2fc319371e1e9b1305c8ec335c0d9c"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64VarSeqQ16Kv128StaticSwapsAbForGen", 81040, 512, 2, 64, 0, 2, 16, 0, true, false, false, false, "d2b31aee1f79831863195e52e0e2ec80270b23c3204a095674b25fca1c21fe08"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen", 81040, 512, 2, 32, 0, 2, 16, 0, true, false, false, false, "cce9c69778438729f3c3f1539e34106adb1d551817b4f3e1c6dd0b577a1a95c0"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16VarSeqQ8Kv128StaticSwapsAbForGen", 141968, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "50772ace457207ce602aa333bb99b51c8ecfa5e20ef4883ca285b1d338ec200c"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64VarSeqQ8Kv128PersistentSwapsAbForGen", 146064, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "49a369ffb904002fc9d21c79b4aa1627ef05a628493c4b9a7078537e0aaec562"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ8Kv128PersistentSwapsAbForGen", 77456, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "187fa62b643af15f11cc3ac0790164c509d9cfe1536b93260a048931e71afe3b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16VarSeqQ16Kv128StaticSwapsAbForGen", 81040, 512, 2, 16, 0, 2, 16, 0, true, false, false, false, "5df1c65f56829ba46f7bce2ec1a2f991c11f4fdce805179360dd1215f5b6bc6d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16VarSeqQ8Kv128PersistentSwapsAbForGen", 77456, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "40ab205997ad9c47f7dc4b95eb57e111d33343a35e051dd9175b6eb6c34227e9"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64VarSeqQ16Kv128StaticSwapsAbForGen", 148624, 512, 2, 64, 0, 2, 16, 0, true, false, false, false, "5364e3f099a30dd6c72a95fb550087d43948db3a02849db8824dd6f25f6c0f59"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64VarSeqQ8Kv128PersistentSwapsAbForGen", 77456, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "365c28fe46d0110b51a325050cf4c067616ba15d5515227cfc2426ec2763e77c"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64VarSeqQ8Kv128StaticSwapsAbForGen", 144016, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "4172147c58639371b63ba574bfc1c89e6cded15876f71ef0b5013bcdbd4d8935"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64VarSeqQ16Kv128PersistentSwapsAbForGen", 85136, 512, 2, 64, 0, 2, 16, 1, true, false, false, false, "efa8dec6e8d024ed47aa1cb82c0aa2b42f129a153dccf7278b7be520d5b44bd4"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen", 148624, 512, 2, 32, 0, 2, 16, 0, true, false, false, false, "9ca43fb972c3fd3c4fd0d246699ac0a6c934d0d5b709cbb73eb0c1aaed5adb0a"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16VarSeqQ8Kv128PersistentSwapsAbForGen", 146064, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "53a0e0e87273b72e3e7dd2abf52ed3c37be953a7952efa0a77a964c2012ba899"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ8Kv128PersistentSwapsAbForGen", 146064, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "0e200e090263ff0901b403df7ee11ff859582c9903bfa5392ed15ea4e7602d66"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16VarSeqQ16Kv128StaticSwapsAbForGen", 148624, 512, 2, 16, 0, 2, 16, 0, true, false, false, false, "ff0417f7ab780e73b0c8210620c2c92bbc618fd9352e7a1b05ebdab5d22948f1"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ16Kv128PersistentSwapsAbForGen", 85136, 512, 2, 32, 0, 2, 16, 1, true, false, false, false, "ab305e14ceff398c12e0eecef76c57bf2c0376fff1db500fa3e9ae96fb26f96b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32VarSeqQ8Kv128StaticSwapsAbForGen", 144016, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "e03b920d53368d5d2afede99920bfed7ac9ea18449b42e7b493cc9f86924aa6d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16VarSeqQ16Kv128PersistentSwapsAbForGen", 85136, 512, 2, 16, 0, 2, 16, 1, true, false, false, false, "049992676a7fff674c3d745d72cecd703a064e44e63535966c513f10923fa5bd"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64VarSeqQ16Kv128PersistentSwapsAbForGen", 156816, 512, 2, 64, 0, 2, 16, 1, true, false, false, false, "3f73b5072efaa02150a4550484e059f8ecbf06d3165afd7cde59a875b9e2cb0d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ16Kv128PersistentSwapsAbForGen", 156816, 512, 2, 32, 0, 2, 16, 1, true, false, false, false, "a25bf73cb4b373c71f635bd3722c9b00f07beb586fd7f1d0187e6bbddab1ecc6"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16VarSeqQ8Kv128StaticSwapsAbForGen", 144016, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "b52183f5ecc44f009fa41ada9509b00efab1d6007c0a16d57607254795ab4530"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16VarSeqQ16Kv128PersistentSwapsAbForGen", 156816, 512, 2, 16, 0, 2, 16, 1, true, false, false, false, "cb1cec34705ed46efa6677a0b933505af6b35f9d34868226a40165acc5858571"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64VarSeqQ8Kv128PersistentSwapsAbForGen", 150160, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "efcdd72686a5107a9b43a886d9af17c609df542c86a36fde37359f2d074657b2"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32VarSeqQ16Kv128StaticSwapsAbForGen", 152720, 512, 2, 32, 0, 2, 16, 0, true, false, false, false, "f948bd80f3b2c0a3890895302676932d954510a4db44012ce760f47da0aaf0d4"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32VarSeqQ8Kv128PersistentSwapsAbForGen", 150160, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "d90445e90c17ba7113fbdf96cde3fb699bfbdcbdf048ce6fe2bc4cc931cb109b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16VarSeqQ16Kv128StaticSwapsAbForGen", 152720, 512, 2, 16, 0, 2, 16, 0, true, false, false, false, "55144b3e0c0485c7ca4194665d01657b9487522fbc3128d10f8aff515a1cf7ad"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16VarSeqQ8Kv128PersistentSwapsAbForGen", 150160, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "7a1b41aac5affe25e31065877e5023282eadd522cbed3eff261e82432912584c"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 75408, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "87994292bf7b8fa560177b596465b59b6626077331c10d833a190265ee235001"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 75408, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "b9ef87d8fa2c38291fa5e028bffb7af648719a0fcb4329473b3dec5e186db965"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 75408, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "b41f1a9a0e1b6054749c989a0be8fe909e211874910b8fe5d07a514315f94421"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 141968, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "3cfe008ed38aeec9f94a87441e47cfc34469d269506fea60d606d1f06b864e7b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 81040, 512, 2, 32, 0, 2, 16, 0, true, true, false, false, "a28a04c30baebf2b962c3ec8c8733df9647440a92a237ad4a3bcf5bda3b0ac4f"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 81040, 512, 2, 16, 0, 2, 16, 0, true, true, false, false, "98ff4b54efac74489624f5fa90cad9ec668554fbdd917913b1c4dd30fec60b02"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 141968, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "808cd0a1415b6f067e725aa1719ea7b4b22edc2478325c5479fb561836f3d936"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 110288, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "258c2ae13d71b5607f297e4185853d5c5d55a75cdb933c52eb403b5ae252c1f4"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 81040, 512, 2, 64, 0, 2, 16, 0, true, true, false, false, "24dd9f61ca605233b107f0d3182c3d48a279acc9b08f0a43530b082b93ad6953"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 141968, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "6948e1a01a4afc099b03257e1bfb3315e9707f051e897610420b5ec3f62bb4bd"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64VarSeqQ16Kv128StaticSwapsAbForGen", 152720, 512, 2, 64, 0, 2, 16, 0, true, false, false, false, "a354a09ab8f8f2e6b43507c9a26820b5839d6d6f3ebb42d75ce754be12a3682d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 110288, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "c3d3d9101752d17906a246c0a83d5f132bb109bd66fcf0014bfbcbfae24886ad"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 110288, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "071fbb223c690f14a8732d4715083f4d081adcc14c4632f1ae03764429380ea6"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 148624, 512, 2, 32, 0, 2, 16, 0, true, true, false, false, "654ea3914daecbcf50720799a875af1ca043ceec2b9fbc6a10a1b4487d6122cc"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 148624, 512, 2, 16, 0, 2, 16, 0, true, true, false, false, "e370e417cb64e01a72b54846666ef9a6ad473dff7ae766f91ddae9685210171a"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 148624, 512, 2, 64, 0, 2, 16, 0, true, true, false, false, "27268a170c859aaefcafadef1dc66d23bc519bd452f55bf7dd43968d47a09964"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 115920, 512, 2, 32, 0, 2, 16, 0, true, true, false, false, "13df85bf82169116d29a495d55b7d5897f0def059b286a2bb6e8f035a5271b5e"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 115920, 512, 2, 64, 0, 2, 16, 0, true, true, false, false, "bc071d91b692fa8e14594b572a8d600d62723ba84128cb78db0e7ecababe5bdf"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 115920, 512, 2, 16, 0, 2, 16, 0, true, true, false, false, "d5d775d3dd7071adbf52a300435b3befbd7221961eaa7ea5d9832296869733ff"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 175824, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "85867c993e63687bc77e6688def141a54b9764c21ed9a97353c6fd2dffe44f65"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 175824, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "d50a190f84cd5bbb0221f5b170217b490503f5ddb33cea57fb8e6d5946a36b4b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 175824, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "d36d81a1425fc7d0deb716d2a7326ed5d4556aa817a72dd289225ab076cecdd4"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64VarSeqQ16Kv128PersistentSwapsAbForGen", 165008, 512, 2, 64, 0, 2, 16, 1, true, false, false, false, "616f8c96a7ce58508a252a20b90b5eb2f71c5ca0f84739217ab4ba0a72646a5e"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32VarSeqQ16Kv128PersistentSwapsAbForGen", 165008, 512, 2, 32, 0, 2, 16, 1, true, false, false, false, "a371a068bc29d693b45dd8b0572fefbc5f25f4ca14461db0a60365a7a2b15ee8"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 144016, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "6e0b371067bec9d1cf5962f0d3159f3fc590420be03b37724e44f0766adf0e39"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 144016, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "43121afe06785e47e211e5ec2f8e75bf88e1a6a15938726bd3ea971a69b19f3d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen", 144016, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "d384978d44ea4bf6f81d8f7fd08edfd51021e76c6120c2472268fee30d1b96af"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16VarSeqQ16Kv128PersistentSwapsAbForGen", 165008, 512, 2, 16, 0, 2, 16, 1, true, false, false, false, "5d3313fac11f48e80aa43cca43ae3ea5de14f69231cbb305d26d7b37c57c613a"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 152720, 512, 2, 64, 0, 2, 16, 0, true, true, false, false, "92ff29bde09467f960f4c5730728d45e0c8a55f3a9e1ace815cbcd1520904c4b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 182480, 512, 2, 16, 0, 2, 16, 0, true, true, false, false, "198fdc59409dc47715dc5f3b3c411158ede9245f579617b802edddd93c94f685"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 152720, 512, 2, 32, 0, 2, 16, 0, true, true, false, false, "95b8b6531c1e3576ff01bec2ee5ce2c0a605a4cca545b10d334ca97fe4b3c794"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen", 152720, 512, 2, 16, 0, 2, 16, 0, true, true, false, false, "46795f71eee402075fd6364b5e504a309c88a65ffe0490fa831105991db119c0"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 182480, 512, 2, 32, 0, 2, 16, 0, true, true, false, false, "2481083dcd31b0831623d6c6b6fb4c0335cbfe852361bcfecedd559c995e8e6a"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 182480, 512, 2, 64, 0, 2, 16, 0, true, true, false, false, "ce7d79157c2125a804fde242a2021e4bc29d32fa892ad8370fec973f0089bdcf"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 177360, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "8cb612d2157371edf1768ba8c34a4fcbc290fcbc69142a8a9f93f2a22695125d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 177360, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "d96b5056fc949f9186e3eef3177d8f26379922a648ac7ec58908bdd7a8719839"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 8, 128, 8, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16MultiCtasKvCgaVarSeqQ8Kv128StaticSwapsAbForGen", 177360, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "16f41e8c713c0e3e04cfcd0f6af31556b4dcc91750145584be8e33d62dd4a20e"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP64MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 186064, 512, 2, 64, 0, 2, 16, 0, true, true, false, false, "e35e5f2778935f50e243197f7eb97eca43ad8a210dca262ebf24fe41013452b2"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP32MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 186064, 512, 2, 32, 0, 2, 16, 0, true, true, false, false, "1868d32ec6c938ec010ab12323fdfbbadf57fb157cb9f097e636ade4a98b444e"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 16, 128, 16, 256, 256, 256, 256, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H256PagedKvDenseP16MultiCtasKvCgaVarSeqQ16Kv128StaticSwapsAbForGen", 186064, 512, 2, 16, 0, 2, 16, 0, true, true, false, false, "75615fcc5ffdd96d4062b2cf6b4a88e578b49389eb8e9403f985f78ea24ff66d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ128Kv128StaticKeepsAbForGen", 86032, 512, 2, 32, 0, 3, 128, 0, true, false, false, false, "ca34bc4fbfec27ce1ac21796fa5bec2c8023fdd9f0c7fc7002e2bd4aa6792580"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64VarSeqQ128Kv128PersistentKeepsAbForGen", 102416, 512, 2, 64, 0, 3, 128, 1, true, false, false, false, "f236d87d00417975f5a2ef2de5e38a1b61db6d4bccf644a6264b08f82f00d6da"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16VarSeqQ128Kv128StaticKeepsAbForGen", 86032, 512, 2, 16, 0, 3, 128, 0, true, false, false, false, "8291568600086b9b139e4687b67ad18b565c33756122a904209bab577f930fef"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16VarSeqQ128Kv128PersistentKeepsAbForGen", 102416, 512, 2, 16, 0, 3, 128, 1, true, false, false, false, "985abdc3e0e34789adc8cd64491325c795cb9bcb9aac76fa7a22959c6c73e22f"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64VarSeqQ128Kv128StaticKeepsAbForGen", 86032, 512, 2, 64, 0, 3, 128, 0, true, false, false, false, "de3f480a71f087baa1941964df77f8d3bfe90ec21427dc8cff508d0fcc8aea4d"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32VarSeqQ128Kv128PersistentKeepsAbForGen", 102416, 512, 2, 32, 0, 3, 128, 1, true, false, false, false, "e93bec95ac4a360c71a2e93a5247fa748678128a069b3577df2410c3b7e6d8a3"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64VarSeqQ128Kv128StaticKeepsAbForGen", 167952, 512, 2, 64, 0, 3, 128, 0, true, false, false, false, "9314df19786c881f10a94712d480fd9423a94a7a4d07a89f8aaa3f2bdd46dfc4"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ128Kv128PersistentKeepsAbForGen", 200720, 512, 2, 32, 0, 3, 128, 1, true, false, false, false, "8067794fb862eb0010465e5c931e47b8a60a510818b5ad1a4a4196cc20bb2fed"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32MultiCtasKvVarSeqQ128Kv128StaticKeepsAbForGen", 167968, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "653bb48afe5e27afcc1a86ae19f0d737edb61464d87cbf92ee7d3bbc46608162"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16VarSeqQ128Kv128PersistentKeepsAbForGen", 200720, 512, 2, 16, 0, 3, 128, 1, true, false, false, false, "5046497401ef38e69d9f3065e9fdf6aba429665822512abc9af252574efed3a3"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16MultiCtasKvVarSeqQ128Kv128StaticKeepsAbForGen", 86048, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "91df47ac7910ca4f235ad7ad592a8cd535908db25d3741bc81cab76b364c4660"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32MultiCtasKvCgaVarSeqQ128Kv128StaticKeepsAbForGen", 120912, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "11b31549f2971642dc8e976b255a8b4b6bbffc16b051f1469846508b4c8d5bcc"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64MultiCtasKvVarSeqQ128Kv128StaticKeepsAbForGen", 86048, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "d1edeb11d5fe9e79e61c1cf88666663796ada8676fe93f1da9b299a13998355e"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16MultiCtasKvVarSeqQ128Kv128StaticKeepsAbForGen", 167968, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "6d6d56f7198ba8a4e7a20471fa4d7d44344a4c4542606f4de49131616be19857"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16VarSeqQ128Kv128StaticKeepsAbForGen", 167952, 512, 2, 16, 0, 3, 128, 0, true, false, false, false, "d9606ef56c79527fdd7100a32d2391ec624cf9cb25683e499a9a0cfc16bb4ffc"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64VarSeqQ128Kv128PersistentKeepsAbForGen", 200720, 512, 2, 64, 0, 3, 128, 1, true, false, false, false, "7e6f7d9c48cdb512b6694360e70e083caabb361db98414c6da61157f3437db3b"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64MultiCtasKvVarSeqQ128Kv128StaticKeepsAbForGen", 167968, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "c815e9b7f457ec714600ac08054bb6b39c140963e38d634c52bf5cf34b87931c"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ128Kv128StaticKeepsAbForGen", 167952, 512, 2, 32, 0, 3, 128, 0, true, false, false, false, "72af2727f29d642cffeb0d1bf5d83d0d3c837104e9739997c891889ed3dded30"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP16MultiCtasKvCgaVarSeqQ128Kv128StaticKeepsAbForGen", 231376, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "a63869737930660230ac872f68e6087b0c27ee931c57298d45f20ac677f26a37"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP32MultiCtasKvVarSeqQ128Kv128StaticKeepsAbForGen", 86048, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "0a25c6c1004761153a792665a573fb05ba327ec8475903441c870428c6f487ac"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP64MultiCtasKvCgaVarSeqQ128Kv128StaticKeepsAbForGen", 120912, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "eeeaf414b195d56f8e48e3744b83ae96b07bf0e17c245575381d406adf3df68f"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 64, 64, 64, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H64PagedKvDenseP16MultiCtasKvCgaVarSeqQ128Kv128StaticKeepsAbForGen", 120912, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "a7bba43c32c69c16257e4aaef211aa28be342e0b67e2d83c63bd3dc4121156df"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP32MultiCtasKvCgaVarSeqQ128Kv128StaticKeepsAbForGen", 231376, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "671b85e18c2f7c0e95d7f7c0278477021c6f77f1ff6426f1bd292ebb5c9c79f9"},
{ DATA_TYPE_BF16, DATA_TYPE_BF16, DATA_TYPE_BF16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QkvBfloat16OBfloat16H128PagedKvDenseP64MultiCtasKvCgaVarSeqQ128Kv128StaticKeepsAbForGen", 231376, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "a1c283b1580f3af7aab8e44592281e99372e57e72a62d79d9543c6a97b6f50e1"},
	};
