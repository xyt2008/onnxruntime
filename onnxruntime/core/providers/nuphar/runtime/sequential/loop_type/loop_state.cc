// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop_type/loop_state.h"

#include "core/providers/nuphar/runtime/sequential/loop_type/lstm_state.h"
#include "core/providers/nuphar/runtime/sequential/loop_type/scan_state.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace tvm_codegen {

std::unique_ptr<DLLoopState> DLLoopState::MakeDLLoopState(const NupharCodeGenCtx& ctx_codegen,
                                                          const LoopOpKind& loop_op_kind,
                                                          const Node& node,
                                                          const LoopDirection& direction) {
  if (loop_op_kind == LoopOpKind::kLoopOpLSTM) {
    ORT_NOT_IMPLEMENTED("not implemented type for LSTM node");
  } else if (loop_op_kind == LoopOpKind::kLoopOpScan) {
    return std::make_unique<DLScanState>(
        ctx_codegen, node, direction, 1);
  }

  // TODO: more ops
  ORT_NOT_IMPLEMENTED("not implemented op kind");
}

void DLLoopState::InitializeInStates(OpKernelContext* context,
                                     const std::vector<int64_t>& initializer_indices,
                                     const std::vector<int64_t>& initializer_sizes,
                                     const std::vector<void*>& target_state_ptrs,
                                     int64_t size_of_tensor,
                                     int64_t input_offset) {
  // indices of initializers. If a state doesn't have initializer.
  // Note that a state still has an initializer if the initializer optional
  // but not provided. We will need this information to assign default value
  // to the state.

  int64_t num_indices = gsl::narrow_cast<int64_t>(initializer_indices.size());
  auto const& args = node_.InputDefs();
  int64_t num_args = gsl::narrow_cast<int64_t>(args.size());
  ORT_UNUSED_PARAMETER(num_args);
  for (int64_t i = 0; i < num_indices; ++i) {
    auto idx = initializer_indices[i];
    ORT_ENFORCE_DEBUG(idx < num_args);
    void* p_data = target_state_ptrs[i];
    int seq_sz = initializer_sizes[i];

    auto const* def = args[idx];
    if (def->Exists()) {
      const Tensor* input_tensor = context->Input<Tensor>(idx);
      if (IsSecondReverseLoop()) {
        ORT_ENFORCE_DEBUG(2 * seq_sz == gsl::narrow_cast<int64_t>(input_tensor->Shape().Size()));
      } else {
        ORT_ENFORCE_DEBUG(seq_sz == gsl::narrow_cast<int64_t>(input_tensor->Shape().Size()));
      }

      const void* input_data = static_cast<const char*>(input_tensor->DataRaw()) + input_offset * size_of_tensor;

      memcpy(p_data, input_data, seq_sz * size_of_tensor);
    } else {
      // default is 0 for LSTM.
      // ISSUE: If any states of other OPs have non-zero default values,
      // then we will need to make corresponding changes here.
      memset(p_data, 0, seq_sz * size_of_tensor);
    }
  }
}

void DLLoopState::InitializeOneTVMArg(PackedFuncCallCtx& ctx_call,
                                      int& arg_idx,
                                      const std::vector<int64_t>& shape,
                                      DLDataType dtype,
                                      const void* p_data) {
  ctx_call.shapes[arg_idx] = shape;
  ctx_call.tvm_tensors[arg_idx] =
      {const_cast<void*>(p_data), ctx_call.tvm_ctx,
       gsl::narrow_cast<int>(ctx_call.shapes[arg_idx].size()), dtype,
       ctx_call.shapes[arg_idx].data(), /*strides*/ nullptr, /*byte_offsets*/ 0};
  ctx_call.lvalues[arg_idx].v_handle = &(ctx_call.tvm_tensors[arg_idx]);

  arg_idx++;
}

std::vector<int64_t> DLLoopState::GetDLTensorShape(const Tensor* tensor,
                                                   const std::vector<int64_t>& expected_shape,
                                                   int slice_start) {
  const TensorShape& shape = tensor->Shape();
  std::vector<int64_t> dl_shape = slice_start > 0 ? shape.Slice(slice_start).GetDims() : shape.GetDims();
  ORT_UNUSED_PARAMETER(expected_shape);
  ORT_ENFORCE_DEBUG(dl_shape == expected_shape);
  return dl_shape;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
