// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop.h"

#include "core/codegen/common/profile.h"

namespace onnxruntime {
namespace tvm_codegen {

// TODO: remove node, after spliting shape update and pointer update
// TODO: remove CodeGenContext after splitting codegen and runtime
void LoopExecBlock::Run(NupharComputeCtx* compute_ctx,
                        PackedFuncCallCtx* ctx_call,
                        const Node& node,
                        const NupharCodeGenCtx& ctx_codegen) {
  if (ctx_call->HasInitialized()) {
    UpdateContext(compute_ctx, ctx_call, node, ctx_codegen);
  } else {
    InitContext(compute_ctx, ctx_call, node, ctx_codegen);
  }

  const PackedFuncInfo& info = ctx_call->info;
  const tvm::runtime::PackedFunc& func = info.packed_func;
  int num_args = info.num_args;

  // TODO: Do it sequentially and worry about parallelization later.
  // TODO: parallelization will happen in another kind of ExecBlock
  while (dl_loop_state_->IsValid()) {
    dl_loop_state_->FillTVMArgs(*ctx_call);
    tvm::TVMArgs tvm_args(ctx_call->lvalues.data(), info.type_codes.data(), num_args);
    tvm::TVMRetValue rvalue;

    {
      CODEGEN_PROFILER_EVENT(loop_CallPacked);
      func.CallPacked(tvm_args, &rvalue);
    }

    dl_loop_state_->Advance();
  }
  dl_loop_state_->LoopFinalize();
}

// TODO: remove node, after spliting shape update and pointer update
// TODO: remove CodeGenContext after splitting codegen and runtime
void LoopExecBlock::InitContext(NupharComputeCtx* compute_ctx,
                                PackedFuncCallCtx* ctx_call,
                                const Node& node,
                                const NupharCodeGenCtx& ctx_codegen) {
  dl_loop_state_ = DLLoopState::MakeDLLoopState(ctx_codegen,  // remove usage of ctx_codegen
                                                loop_kind_,
                                                node,  // remove usage of node
                                                direction_);

  const PackedFuncInfo& info = ctx_call->info;

  size_t input_count = info.input_count;
  size_t num_actual_outputs = info.num_actual_outputs;

  size_t num_args = input_count + num_actual_outputs;
  auto& lvalues = ctx_call->lvalues;
  lvalues.resize(num_args);
  auto& tvm_tensors = ctx_call->tvm_tensors;
  tvm_tensors.resize(num_args);
  auto& shapes = ctx_call->shapes;
  shapes.resize(num_args);

  dl_loop_state_->InitContext(compute_ctx->GetOpKernelContext(), *ctx_call);
}

// TODO: remove node, after spliting shape update and pointer update
// TODO: remove CodeGenContext after splitting codegen and runtime
void LoopExecBlock::UpdateContext(NupharComputeCtx* compute_ctx,
                                  PackedFuncCallCtx* ctx_call,
                                  const Node& /*node*/,
                                  const NupharCodeGenCtx& /*ctx_codegen*/) {
  dl_loop_state_->UpdateContext(compute_ctx->GetOpKernelContext(), *ctx_call);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
