// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/runtime/sequential/loop_type/loop_state.h"

namespace onnxruntime {
namespace tvm_codegen {

//dummpy one
class DLLSTMState {
 public:
  DLLSTMState();
};

// disable LSTM state
// leave the below in comment for later rewriting
// TODO: Rewrite LSTM state
#if 0
// T represent input/output tensor types such as float, double, etc.
template <typename T>
class DLLSTMState : public DLLoopState {
 public:
  // OLD one
  DLLSTMState(OpKernelContext* context,
              CodeGenContext& meta,
              //std::unordered_map<std::string, int64_t>& realized_dims,
              const LoopOpKind& loop_op_kind,
              const Node& node,
              const tvm::Array<tvm::Tensor>& in_state_tensors,
              const tvm::Array<tvm::Tensor>& out_state_tensors,
              const LoopDirection& direction,
              int num_directions)
      : DLLoopState(meta, /*realized_dims,*/ loop_op_kind, node, in_state_tensors, out_state_tensors, direction, num_directions),
        context_(context),
        H_in_tensor_(in_state_tensors[0]),
        H_out_tensor_(out_state_tensors[0]),
        C_in_tensor_(in_state_tensors[1]),
        C_out_tensor_(out_state_tensors[1]),
        size_of_tensor_elem_(0),
        hidden_size_(0),
        seq_length_(0),
        batch_size_(0),
        input_size_(0),
        input_ptr_(nullptr),
        H_in_ptr_(nullptr),
        H_out_ptr_(nullptr),
        C_in_ptr_(nullptr),
        C_out_ptr_(nullptr),
        Y_ptr_(nullptr),
        Y_h_ptr_(nullptr),
        Y_c_ptr_(nullptr),
        output_step_stride_(0){};

  virtual ~DLLSTMState() = default;

  virtual int GetMinSequenceLen() override {
    return std::min(seq_length_, DLLoopState::GetMinSequenceLen());
  }

  virtual Status LoopInitialize(OpKernelContext* context, LoopPackedFuncCtxOld& packed_func_ctx) override;

  virtual void LoopInitializeWithReuse(OpKernelContext* context, LoopPackedFuncCtxOld& packed_func_ctx) override {
    LoopInitialize(context, packed_func_ctx);
    current_loop_step_ = 0;
  }

  virtual void LoopFinalize() override;

  virtual void Advance() override;

  virtual void FillTVMArgs(LoopPackedFuncCtxOld& packed_func_ctx) override;

 private:
  void AllocateStatesData();

  void InitializeSequenceLens();

  void InitializeTVMArgs(LoopPackedFuncCtxOld& packed_func_ctx);

  void InitializeOneStateArg(LoopPackedFuncCtxOld& packed_func_ctx,
                             int& arg_idx,
                             const std::vector<int64_t>& shape);

  void InitializeOneInputArg(LoopPackedFuncCtxOld& packed_func_ctx,
                             int input_idx,
                             int& arg_idx,
                             const std::vector<int64_t>& expected_shape,
                             DLDataType dtype,
                             int slice_start,
                             std::vector<int>& initializer_slice_starts);

  void InitializeOneInitializer(LoopPackedFuncCtxOld& packed_func_ctx,
                                int& arg_idx,
                                const Tensor* input_tensor,
                                int slice_start);

  void InitializeOneOutputArg(LoopPackedFuncCtxOld& packed_func_ctx,
                              int output_idx,
                              int& arg_idx,
                              const std::vector<int64_t>& expected_shape,
                              DLDataType dtype,
                              int slice_start);

  int64_t GetSequenceOffset() {
    if (IsSecondReverseLoop())
      return batch_size_ * hidden_size_;
    else
      return 0;
  }

  bool HasOutputDef(int idx) {
    return idx < static_cast<int>(node_.OutputDefs().size());
  }

  OpKernelContext* context_;

  tvm::Tensor H_in_tensor_;
  tvm::Tensor H_out_tensor_;
  tvm::Tensor C_in_tensor_;
  tvm::Tensor C_out_tensor_;

  // tvm dl type
  DLDataType input_dl_type_;
  int64_t size_of_tensor_elem_;

  int64_t hidden_size_;
  // these values will be initialized with the X input
  int seq_length_;
  int64_t batch_size_;
  int64_t input_size_;

  // points to the current head of X
  const T* input_ptr_;
  // points to the memory allocated for reversed inputs in reverse direction
  IAllocatorUniquePtr<T> allocated_reverse_input_ptr_;

  IAllocatorUniquePtr<T> allocated_H_in_ptr_;
  T* H_in_ptr_;
  IAllocatorUniquePtr<T> allocated_H_out_ptr_;
  T* H_out_ptr_;

  IAllocatorUniquePtr<T> allocated_C_in_ptr_;
  T* C_in_ptr_;
  IAllocatorUniquePtr<T> allocated_C_out_ptr_;
  T* C_out_ptr_;

  // Y
  IAllocatorUniquePtr<T> allocated_Y_ptr_;
  T* Y_ptr_;
  // Y_h
  T* Y_h_ptr_;
  // Y_c
  T* Y_c_ptr_;

  // point to dummy output tensors' buffers.
  // Although they are not used in actual computation, we still need memory space for holding
  // dummy zero tensors.
  std::vector<IAllocatorUniquePtr<T>> dummy_output_ptrs_;

  int64_t output_step_stride_;
};
#endif

}  // namespace tvm_codegen
}  // namespace onnxruntime
