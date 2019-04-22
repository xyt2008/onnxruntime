// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// BasicExecBlock is most common execuction block
// It does not contain C++ control flow.
class BasicExecBlock : public ExecBlock {
 public:
  BasicExecBlock(const std::string& name)
      : ExecBlock(name, "BasicExecBlock") {}

  // TODO: remove node, after spliting shape update and pointer update
  // TODO: remove CodeGenContext after splitting codegen and runtime
  void Run(NupharComputeCtx* compute_ctx,
           PackedFuncCallCtx* ctx_call,
           const Node& node,
           const NupharCodeGenCtx& ctx_codegen) override;
  void InitContext(NupharComputeCtx* compute_ctx,
                   PackedFuncCallCtx* ctx_call,
                   const Node& node,
                   const NupharCodeGenCtx& ctx_codegen) override;
  void UpdateContext(NupharComputeCtx* compute_ctx,
                     PackedFuncCallCtx* ctx_call,
                     const Node& node,
                     const NupharCodeGenCtx& ctx_codegen) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BasicExecBlock);

  enum : int {
    OutputAliased = -1,
  };
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
