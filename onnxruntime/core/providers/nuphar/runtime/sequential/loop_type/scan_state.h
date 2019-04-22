// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/runtime/sequential/loop_type/loop_state.h"

namespace onnxruntime {
namespace tvm_codegen {

class DLScanState final : public DLLoopState {
 public:
  DLScanState(const NupharCodeGenCtx& ctx_codegen,
              const Node& node,
              const LoopDirection& direction,
              int num_directions)
      : DLLoopState(ctx_codegen, node, direction, num_directions) {
    subgraph_ = node.GetGraphAttribute("body");
    max_seq_length_ = 0;
  };

  void InitContext(OpKernelContext* context, PackedFuncCallCtx& packed_func_ctx) override;
  void UpdateContext(OpKernelContext* context, PackedFuncCallCtx& packed_func_ctx) override;
  void FillTVMArgs(PackedFuncCallCtx& packed_func_ctx) override;

  void LoopFinalize() override;
  void Advance() override;

 private:
  //
  std::vector<bool> is_state_alias_;
  std::vector<bool> is_output_alias_;
  std::vector<int> state_to_output_index_;
  std::vector<int> output_to_state_index_;

  // Common
  std::vector<void*> input_ptrs_;
  std::vector<void*> output_ptrs_;
  std::vector<void*> state_output_ptrs_;

  // Buffers
  std::vector<void*> state_input_buffers_;
  std::vector<void*> state_output_buffers_;

  // allocated state buffer
  std::vector<IAllocatorUniquePtr<void>> allocated_input_state_buffers_;
  std::vector<IAllocatorUniquePtr<void>> allocated_output_state_buffers_;

  std::vector<int64_t> input_strides_;
  std::vector<int64_t> output_strides_;

  std::vector<std::size_t> state_bytes_size_;

  // Common
  std::vector<void*> current_state_input_ptrs_;
  std::vector<void*> current_state_output_ptrs_;

  std::vector<std::vector<int64_t>> output_dims_;  // for ORT tensor shapes
  std::vector<bool> input_is_initializer_;

  int64_t max_seq_length_;

  // Scan only
  const Graph* subgraph_;

  int64_t num_scan_inputs_;
  std::vector<int64_t> input_directions_;
  std::vector<int64_t> output_directions_;
  std::vector<int64_t> input_sequence_axes_;

  int64_t num_loop_state_variables_;
  int64_t num_variadic_inputs_;
  int64_t num_variadic_outputs_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
