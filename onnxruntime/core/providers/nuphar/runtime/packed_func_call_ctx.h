// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/compiler/packed_func_ctx.h"
#include <tvm/build_module.h>

namespace onnxruntime {
namespace tvm_codegen {

// PackedFuncCallCtx holds runtime fill-in for tvm::runtime::PackedFunc
// It contains runtime information
// Those static known meta will be put in PackedFuncInfo
struct PackedFuncCallCtx {
  const DLContext& tvm_ctx;
  std::unordered_map<std::string, int64_t>& realized_dims;
  const PackedFuncInfo& info;

  std::vector<TVMValue> lvalues;
  std::vector<DLTensor> tvm_tensors;
  std::vector<std::vector<int64_t>> shapes;

  // map node's input/output index to corresponding tvm_tensors
  std::vector<DLTensor*> context_idx_to_tvm_tensors;

  // TODO: add comment
  struct DimPatch {
    std::pair<size_t, size_t> from_context_input_idx_and_dim;
    std::vector<std::pair<size_t, size_t>> tvm_arg_idx_and_dim;
    std::string dim_param;
  };
  std::vector<DimPatch> dim_patches;

  bool HasInitialized() const {
    return tvm_tensors.size() > 0;
  }

  PackedFuncCallCtx(const DLContext& _tvm_ctx,
                    const PackedFuncInfo& _info,
                    std::unordered_map<std::string, int64_t>& _realized_dims)
      : tvm_ctx(_tvm_ctx),
        realized_dims(_realized_dims),
        info(_info) {}

  // TODO: refactor this, remove template
  // create or update dim patch, note that create dim patch only happens for input
  template <bool is_input>
  void CreateOrUpdateDimPatch(const std::string& dim_param, size_t context_idx, size_t tvm_arg_idx, size_t dim) {
    auto iter = std::find_if(dim_patches.begin(),
                             dim_patches.end(),
                             [&dim_param](const auto& p) {
                               return p.dim_param == dim_param;
                             });
    if (is_input && iter == dim_patches.end()) {
      dim_patches.emplace_back(DimPatch({{context_idx, dim},
                                         {{tvm_arg_idx, dim}},
                                         dim_param}));
    } else {
      ORT_ENFORCE(iter != dim_patches.end());
      auto& dim_patch = *iter;
      dim_patch.tvm_arg_idx_and_dim.emplace_back(std::make_pair(tvm_arg_idx, dim));
    }
  }
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
