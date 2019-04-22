// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/providers/nuphar/runtime/sequential/loop_type/loop_state.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// LoopExecBlock is an ExecBlock for regular loops.
// It is mainly for Scan, LSTM, GRU, RNN, those recurrence.
// It ONLY works a single nested loop.
// It ONLY works a loop body without other controll flow.

class LoopExecBlock : public ExecBlock {
 public:
  LoopExecBlock(LoopDirection direction,
                LoopOpKind kind,
                const std::string& name)
      : ExecBlock(name, "LoopExecBlock"),
        direction_(direction),
        loop_kind_(kind) {}

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
  LoopDirection direction_;
  // TODO refactor DLLoopState
  std::unique_ptr<DLLoopState> dl_loop_state_;
  LoopOpKind loop_kind_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LoopExecBlock);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
