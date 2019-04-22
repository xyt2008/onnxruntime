// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/packed_func_call_ctx.h"
#include "core/providers/nuphar/runtime/sequential/loop_type/loop_utils.h"
#include <tvm/runtime/c_runtime_api.h>

namespace onnxruntime {

class OpKernelContext;

namespace tvm_codegen {

// utilit  function
inline std::size_t GetStaticSize(const Tensor& tensor, size_t start = 0) {
  auto shape = tensor.Shape().GetDims();
  int64_t total_size = gsl::narrow_cast<int64_t>(tensor.DataType()->Size());

  for (size_t i = start; i < shape.size(); i++)
    total_size *= shape[i];

  return total_size;
}

// Base class for holding loop states for running loop-like ops.
class DLLoopState {
 public:
  static std::unique_ptr<DLLoopState> MakeDLLoopState(const NupharCodeGenCtx& ctx_codegen,
                                                      const LoopOpKind& loop_op_kind,
                                                      const Node& node,
                                                      const LoopDirection& direction);

  virtual void InitContext(OpKernelContext* context, PackedFuncCallCtx& packed_func_ctx) = 0;
  virtual void UpdateContext(OpKernelContext* context, PackedFuncCallCtx& packed_func_ctx) = 0;
  virtual void FillTVMArgs(PackedFuncCallCtx& packed_func_ctx) = 0;

  virtual void LoopFinalize() = 0;
  // marching to next loop iteration
  virtual void Advance() = 0;
  virtual bool IsValid() {
    return current_loop_step_ < max_loop_step_;
  }

 protected:
  DLLoopState(
      const NupharCodeGenCtx& ctx_codegen,
      const Node& node,
      const LoopDirection& direction,  // num_directions can be used for computing offset and advance stride in LSTM, GRU, RNN, and Scan
      int num_directions               // num_directions can be used for computing strides in LSTM, GRU, RNN
      )
      : ctx_codegen_(ctx_codegen),
        node_(node),
        current_loop_step_(0),
        min_loop_step_(0),
        max_loop_step_(0),
        direction_(direction),
        num_directions_(num_directions) {}

  // TODO: remove this
  void InitializeInStates(OpKernelContext* context,
                          const std::vector<int64_t>& initializer_indices,
                          const std::vector<int64_t>& initializer_sizes,
                          const std::vector<void*>& target_state_ptrs,
                          int64_t size_of_tensor,
                          int64_t input_offset);

  virtual int GetMaxSequenceLen() {
    return *std::max_element(sequence_lens_.cbegin(), sequence_lens_.cend());
  }

  // ???
  virtual int GetMinSequenceLen() {
    return *std::min_element(sequence_lens_.cbegin(), sequence_lens_.cend());
  }

  void InitializeOneTVMArg(PackedFuncCallCtx& ctx_call,
                           int& arg_idx,
                           const std::vector<int64_t>& shape,
                           DLDataType dtype,
                           const void* p_data = nullptr);

  std::vector<int64_t> GetDLTensorShape(const Tensor* tensor,
                                        const std::vector<int64_t>& expected_shape,
                                        int slice_start);

  bool IsSecondReverseLoop() {
    return num_directions_ == 2 && direction_ == LoopDirection::kReverse;
  }

  /// TODO remove local referene ctx_codegen
  const NupharCodeGenCtx& ctx_codegen_;

  const Node& node_;

  std::vector<int> sequence_lens_;
  // current sequence index that are going to run
  int current_loop_step_;
  int min_loop_step_;
  // the loop terminates when current_idx_ >= max_idx_;
  int max_loop_step_;

  LoopDirection direction_;
  // We need original num_directions for reversing inpus/output and computing output steps
  int num_directions_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
