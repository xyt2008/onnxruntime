// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop_type/lstm_state.h"

#include "core/codegen/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace tvm_codegen {

//dummy one
DLLSTMState::DLLSTMState() {}

// disable LSTM state
// leave the below in comment for later rewriting
// TODO: Rewrite LSTM state
#if 0
template <typename T>
void DLLSTMState<T>::AllocateStatesData() {
  // hidden state tensor and cell state tensor have the same size:
  // batch_size_ * hidden_size_ * size_of_tensor_elem_
  int64_t size_of_state_data = batch_size_ * hidden_size_ * size_of_tensor_elem_;

  // Allocate memory for reverse input
  if (direction_ == LoopDirection::kReverse) {
    allocated_reverse_input_ptr_ = ctx_codegen_.AllocateT<T>(seq_length_ * batch_size_ * input_size_ * size_of_tensor_elem_);
    ORT_ENFORCE_DEBUG(allocated_reverse_input_ptr_ != nullptr, "failed to allocate memory for reverse input!");
  }

  // Later, in InitializeInStates, we will copy initial_h values to H_in_ptr_, which is the first input to LSTM cell.
  allocated_H_in_ptr_ = ctx_codegen_.AllocateT<T>(size_of_state_data);
  H_in_ptr_ = allocated_H_in_ptr_.get();
  ORT_ENFORCE_DEBUG(H_in_ptr_ != nullptr, "failed to allocate memory for H in state data!");
  // If Y is provided, H_out_ptr_ points to the head of Y's data buffer.
  // Otherwise, we create one for it.
  if (node_.OutputDefs()[0]->Exists()) {
    Tensor* output_tensor = context_->Output(0, TensorShape({seq_length_, num_directions_, batch_size_, hidden_size_}));
    // For reverse loop, we allocate memory for reverse loop and then copy the content
    // to original buffer of the output tensor.
    if (direction_ == LoopDirection::kReverse) {
      allocated_Y_ptr_ = ctx_codegen_.AllocateT<T>(seq_length_ * size_of_state_data);
      Y_ptr_ = allocated_Y_ptr_.get();
    } else {
      Y_ptr_ = static_cast<T*>(output_tensor->MutableDataRaw());
    }
    H_out_ptr_ = Y_ptr_;
    // see the comment above the declaration of dummy_output_ptrs_ in lstm_state.h
    dummy_output_ptrs_.push_back(ctx_codegen_.AllocateT<T>(size_of_state_data));
  } else {
    allocated_H_out_ptr_ = ctx_codegen_.AllocateT<T>(size_of_state_data);
    H_out_ptr_ = allocated_H_out_ptr_.get();
    ORT_ENFORCE_DEBUG(H_out_ptr_ != nullptr, "failed to allocate memory for H out state data!");
    dummy_output_ptrs_.push_back(nullptr);
  }

  // buffer for Y_h if it's presented
  if (HasOutputDef(1) && node_.OutputDefs()[1]->Exists()) {
    Tensor* Y_h_tensor = context_->Output(1, TensorShape({num_directions_, batch_size_, hidden_size_}));
    Y_h_ptr_ = static_cast<T*>(Y_h_tensor->MutableDataRaw());
    Y_h_ptr_ += GetSequenceOffset();
    dummy_output_ptrs_.push_back(ctx_codegen_.AllocateT<T>(size_of_state_data));
  } else {
    dummy_output_ptrs_.push_back(nullptr);
  }

  // buffer for Y_c if it's presented
  if (HasOutputDef(1) && node_.OutputDefs()[2]->Exists()) {
    Tensor* Y_c_tensor = context_->Output(2, TensorShape({num_directions_, batch_size_, hidden_size_}));
    Y_c_ptr_ = static_cast<T*>(Y_c_tensor->MutableDataRaw());
    Y_c_ptr_ += GetSequenceOffset();
    dummy_output_ptrs_.push_back(ctx_codegen_.AllocateT<T>(size_of_state_data));
  } else {
    dummy_output_ptrs_.push_back(nullptr);
  }

  // No matter Y_C is given or not, we always need memory locations for Cell states.
  // If Y_C is given, it will get the copy from the last C_out_ptr_
  allocated_C_in_ptr_ = ctx_codegen_.AllocateT<T>(size_of_state_data);
  C_in_ptr_ = allocated_C_in_ptr_.get();
  ORT_ENFORCE_DEBUG(C_in_ptr_ != nullptr, "failed to allocate memory for C_in state data!");
  allocated_C_out_ptr_ = ctx_codegen_.AllocateT<T>(size_of_state_data);
  C_out_ptr_ = allocated_C_out_ptr_.get();
  ORT_ENFORCE_DEBUG(C_out_ptr_ != nullptr, "failed to allocate memory for C_out state data!");
}

// initialize sequence_lens_ with the values of optional sequence_lens if it is presented.
// otherwise initialize all of its elements with seq_length_.
template <typename T>
void DLLSTMState<T>::InitializeSequenceLens() {
  // shape: [batch_size]
  const Tensor* sequence_lens_tensor = context_->Input<Tensor>(4);

  if (sequence_lens_tensor) {
    const auto& seq_shape = sequence_lens_tensor->Shape();
    ORT_ENFORCE(seq_shape.NumDimensions() == 1 && seq_shape[0] == batch_size_);
    const int* p_data = sequence_lens_tensor->Data<int>();
    for (int64_t i = 0; i < batch_size_; ++i, ++p_data) {
      int len = *p_data;
      ORT_ENFORCE(len > 0 || len <= seq_length_);
      sequence_lens_.push_back(len);
    }
  } else {
    sequence_lens_ = std::vector<int>(batch_size_, seq_length_);
  }
}

template <typename T>
void DLLSTMState<T>::Advance() {
  ORT_ENFORCE(current_loop_step_ < max_loop_step_);

#ifdef ENABLE_DUMP_RNN
  std::cout << "Advance: finishing loop step: " << current_loop_step_
            << " (" << LoopDirectionToStr(direction_) << ")\n";
#endif  // ENABLE_DUMP_RNN
  DumpRNNArray("Advance: prev X", input_ptr_, {batch_size_, input_size_});
  DumpRNNArray("Advance: prev H_out", H_out_ptr_, {batch_size_, hidden_size_});
  DumpRNNArray("Advance: prev C_out", C_out_ptr_, {batch_size_, hidden_size_});
  // set corresponding part of Y to 0 if current_loop_step_ >= sequence_length
  if (Y_ptr_) {
    for (int row = 0; row < batch_size_; ++row) {
      if (current_loop_step_ >= min_loop_step_ && current_loop_step_ >= sequence_lens_[row]) {
        auto offset = current_loop_step_ * output_step_stride_ + row * hidden_size_;
        memset(Y_ptr_ + offset, 0, hidden_size_ * size_of_tensor_elem_);
      }
    }
  } else {
    DumpRNNArray("Advance (Y_ptr_ is null): prev Y_h_ptr_ for this direciton", Y_h_ptr_, {batch_size_, hidden_size_});
    for (int row = 0; row < batch_size_; ++row) {
      auto seq_len = sequence_lens_[row];
      if (current_loop_step_ == seq_len - 1) {
        auto offset = row * hidden_size_;
        memcpy(Y_h_ptr_ + offset, H_out_ptr_ + offset, hidden_size_ * size_of_tensor_elem_);
      }
    }
    DumpRNNArray("Advance (Y_ptr is null): curr Y_h_ptr_ for this direction", Y_h_ptr_, {batch_size_, hidden_size_});
  }

  // copy last output of the cell to Y_c
  if (Y_c_ptr_) {
    for (int row = 0; row < batch_size_; ++row) {
      if ((current_loop_step_ + 1) == sequence_lens_[row]) {
        auto offset = row * hidden_size_;
        memcpy(Y_c_ptr_ + offset, C_out_ptr_ + offset, hidden_size_ * size_of_tensor_elem_);
      }
    }
  }

  // move input head
  input_ptr_ += batch_size_ * input_size_;
  DumpRNNArray("Advance: curr X", input_ptr_, {batch_size_, input_size_});

  // move in_state
  // previous H_out_ptr_ becomes current H_in_ptr_
  if (Y_ptr_) {
    H_in_ptr_ = H_out_ptr_;
    H_out_ptr_ += output_step_stride_;
  } else {
    // If Y is not presented, we simply swap H_in and H_out
    std::swap(H_in_ptr_, H_out_ptr_);
  }
  DumpRNNArray("Advance: curr H_in", H_in_ptr_, {batch_size_, hidden_size_});

  // swap Cell state - previous C_out_ptr_ becomes current C_in_ptr_,
  // and previous C_in_ptr_ becomes garbage and will be used for holding current Cell state output.
  std::swap(C_in_ptr_, C_out_ptr_);
  DumpRNNArray("Advance: curr C_in", C_in_ptr_, {batch_size_, hidden_size_});

  // increase loop index
  ++current_loop_step_;
}

template <typename T>
void DLLSTMState<T>::InitializeOneStateArg(LoopPackedFuncCtxOld& packed_func_ctx,
                                           int& arg_idx,
                                           const std::vector<int64_t>& shape) {
  InitializeOneTVMArg(packed_func_ctx, arg_idx, shape, input_dl_type_);
}

template <typename T>
void DLLSTMState<T>::InitializeOneInputArg(LoopPackedFuncCtxOld& packed_func_ctx,
                                           int input_idx,
                                           int& arg_idx,
                                           const std::vector<int64_t>& expected_shape,
                                           DLDataType dtype,
                                           int slice_start,
                                           std::vector<int>& initializer_slice_starts) {
  const NodeArg* def = node_.InputDefs()[input_idx];
  // If this input is provided as an initializer, we skip it.
  // The corresponding initializer will be handled after all non-initializer inputs are processed.
  if (ctx_codegen_.IsInitializerMarshalled(def->Name())) {
    initializer_slice_starts.push_back(slice_start);
    return;
  }
  const Tensor* tensor = context_->Input<Tensor>(input_idx);
  std::vector<int64_t> dl_shape = GetDLTensorShape(tensor, expected_shape, slice_start);
  InitializeOneTVMArg(packed_func_ctx, arg_idx, dl_shape, dtype, tensor->DataRaw());
}

template <typename T>
void DLLSTMState<T>::InitializeOneInitializer(LoopPackedFuncCtxOld& packed_func_ctx,
                                              int& arg_idx,
                                              const Tensor* input_tensor,
                                              int slice_start) {
  const TensorShape& shape = input_tensor->Shape();
  std::vector<int64_t> dl_shape = slice_start > 0 ? shape.Slice(slice_start).GetDims() : shape.GetDims();
  DLDataType dl_type = ToTvmDLDataType(input_tensor->DataType());
  InitializeOneTVMArg(packed_func_ctx, arg_idx, dl_shape, dl_type, input_tensor->DataRaw());
}

template <typename T>
void DLLSTMState<T>::InitializeOneOutputArg(LoopPackedFuncCtxOld& packed_func_ctx,
                                            int output_idx,
                                            int& arg_idx,
                                            const std::vector<int64_t>& expected_shape,
                                            DLDataType dtype,
                                            int slice_start) {
  const Tensor* tensor = context_->Output<Tensor>(output_idx);
  std::vector<int64_t> dl_shape = GetDLTensorShape(tensor, expected_shape, slice_start);
  ORT_ENFORCE(output_idx < gsl::narrow_cast<int>(dummy_output_ptrs_.size()));
  // We bind outputs to dummy pointers, which are not used for computing LSTM cell.
  InitializeOneTVMArg(packed_func_ctx, arg_idx, dl_shape, dtype, dummy_output_ptrs_[output_idx].get());
}

template <typename T>
void DLLSTMState<T>::InitializeTVMArgs(LoopPackedFuncCtxOld& packed_func_ctx) {
  int arg_idx = 0;
  int num_of_in_states = gsl::narrow_cast<int>(in_state_tensors_.size());
  // initialization for H_in
  InitializeOneStateArg(packed_func_ctx, arg_idx, /*shape*/ {batch_size_, hidden_size_});
  // initialization for C_in
  InitializeOneStateArg(packed_func_ctx, arg_idx, /*shape*/ {batch_size_, hidden_size_});
  ORT_ENFORCE(arg_idx == num_of_in_states);

  std::vector<int> initializer_slice_starts;
  int input_idx = 0;
  // initialization for X
  // TODO: input projection needs to be moved to outside of the sequence loop
  InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                        /*expected_shape*/ {batch_size_, input_size_},
                        input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  input_idx++;

  // initialization for W
  InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                        /*expected_shape*/ {4 * hidden_size_, input_size_},
                        input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  input_idx++;

  // initialization for R
  InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                        /*expected_shape*/ {4 * hidden_size_, hidden_size_},
                        input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  input_idx++;

  // initialization for B
  if (node_.InputDefs()[input_idx]->Exists()) {
    InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                          /*expected_shape*/ {8 * hidden_size_},
                          input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  }
  input_idx++;

  // initialization for sequence_lens
  if (node_.InputDefs()[input_idx]->Exists()) {
    // Note that this input is actually not used by our LSTM cell tensor.
    DLDataType seq_length_dl_type =
        ToTvmDLDataType(context_->Input<Tensor>(input_idx)->DataType());
    InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                          /*expected_shape*/ {batch_size_},
                          seq_length_dl_type, /*slice_start*/ 0, initializer_slice_starts);
  }
  input_idx++;

  // initialization for initial_h
  if (node_.InputDefs()[input_idx]->Exists()) {
    // Note that this input is actually not used by our LSTM cell tensor.
    InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                          /*expected_shape*/ {batch_size_, hidden_size_},
                          input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  }
  input_idx++;

  // initialization for initial_c
  if (node_.InputDefs()[input_idx]->Exists()) {
    // Note that this input is actually not used by our LSTM cell tensor.
    InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                          /*expected_shape*/ {batch_size_, hidden_size_},
                          input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  }
  input_idx++;

  // initialization for P
  if (node_.InputDefs()[input_idx]->Exists()) {
    InitializeOneInputArg(packed_func_ctx, input_idx, arg_idx,
                          /*expected_shape*/ {3 * hidden_size_},
                          input_dl_type_, /*slice_start*/ 1, initializer_slice_starts);
  }
  input_idx++;

  ORT_ENFORCE(input_idx == node_.InputDefs().size());

  // Now we start initializing initializers
  int initializer_idx = 0;
  int num_initializers = gsl::narrow_cast<int>(initializer_slice_starts.size());

  for (const auto& item : ctx_codegen_.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      bool is_marshalled = info.layout_info->is_marshalled;
      ORT_ENFORCE(initializer_idx < num_initializers);
      InitializeOneInitializer(packed_func_ctx, arg_idx,
                               is_marshalled ? info.layout_info->marshalled_initializer.get()
                                             : info.original_initializer,
                               initializer_slice_starts[initializer_idx++]);
    }
  }
  ORT_ENFORCE(initializer_idx == num_initializers);

  ORT_ENFORCE(arg_idx == packed_func_ctx.existing_input_count + num_of_in_states);

  DumpRNNArray("W", static_cast<const T*>(packed_func_ctx.tvm_tensors[num_of_in_states + 1].data),
               {4 * hidden_size_, input_size_});
  DumpRNNArray("R", static_cast<const T*>(packed_func_ctx.tvm_tensors[num_of_in_states + 2].data),
               {4 * hidden_size_, hidden_size_});

  // initialization for H_out
  InitializeOneStateArg(packed_func_ctx, arg_idx, /*shape*/ {batch_size_, hidden_size_});
  // initialization for C_out
  InitializeOneStateArg(packed_func_ctx, arg_idx, /*shape*/ {batch_size_, hidden_size_});

  // Note that although we don't use outputs for actual LSTM cell computation,
  // we still need to initialize these output args to make complete TVM args.
  // If we didn't push these dummy outputs into tvm_args_ while building tvm tensors,
  // we wouldn't need the initializations below. The rationale was that we
  // prefer tvm_state::Build to being general enough while avoiding any special
  // routines for LSTM.
  int output_idx = 0;
  int existing_output_cnt = 0;
  // initialization for outputs[0]
  if (HasOutputDef(output_idx)) {
    if (node_.OutputDefs()[output_idx]->Exists()) {
      InitializeOneOutputArg(packed_func_ctx, output_idx, arg_idx,
                             /*expected_shape*/ {batch_size_, hidden_size_},
                             input_dl_type_, /*slice_start*/ 2);
      existing_output_cnt++;
    }
    output_idx++;
  }

  // initialization for outputs[1]
  if (HasOutputDef(output_idx)) {
    if (node_.OutputDefs()[output_idx]->Exists()) {
      InitializeOneOutputArg(packed_func_ctx, output_idx, arg_idx,
                             /*expected_shape*/ {batch_size_, hidden_size_},
                             input_dl_type_, /*slice_start*/ 1);
      existing_output_cnt++;
    }
    output_idx++;
  }

  // initialization for outputs[2]
  if (HasOutputDef(output_idx)) {
    if (node_.OutputDefs()[output_idx]->Exists()) {
      InitializeOneOutputArg(packed_func_ctx, output_idx, arg_idx,
                             /*expected_shape*/ {batch_size_, hidden_size_},
                             input_dl_type_, /*slice_start*/ 1);
      existing_output_cnt++;
    }
    output_idx++;
  }

  ORT_ENFORCE(existing_output_cnt == packed_func_ctx.existing_output_count);
  ORT_ENFORCE(arg_idx == packed_func_ctx.existing_input_count +
                             2 * num_of_in_states + existing_output_cnt);
}

template <typename T>
Status DLLSTMState<T>::LoopInitialize(OpKernelContext* context, LoopPackedFuncCtxOld& packed_func_ctx) {
  context_ = context;

  ProtoHelperNodeContext ctx(node_);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
  ORT_ENFORCE(attrs.GetAttr("hidden_size", &hidden_size_).IsOK());

  // shape: [seq_length, batch_size, input_size]
  const Tensor& X = *context_->Input<Tensor>(0);
  input_dl_type_ = ToTvmDLDataType(X.DataType());
  size_of_tensor_elem_ = gsl::narrow_cast<int64_t>(H_in_tensor_->dtype.bytes());

  auto& X_shape = X.Shape();
  seq_length_ = gsl::narrow<int>(X_shape[0]);
  batch_size_ = gsl::narrow<int>(X_shape[1]);
  input_size_ = gsl::narrow<int>(X_shape[2]);

  AllocateStatesData();
  InitializeSequenceLens();

  // Setup loop conditions
  min_loop_step_ = GetMinSequenceLen();
  max_loop_step_ = GetMaxSequenceLen();

  // We take a similar approach that we used in deep_cpu_lstm:
  // The Y output has shape of [seq,num_direction,batch_size,hidden_size].
  // For bidirectional LSTM, the output of forward loop is placed into
  // [seq,0,batch_size,hidden_size], and the output of reverse loop is placed into
  // [seq,1,batch_size,hidden_size], respectively.
  // This approach allows us to write the output of forward loop directly
  if (num_directions_ == 2 && direction_ == LoopDirection::kForward) {
    output_step_stride_ = 2 * batch_size_ * hidden_size_;
  } else {
    output_step_stride_ = batch_size_ * hidden_size_;
  }

  if (direction_ == LoopDirection::kReverse) {
    ORT_ENFORCE(allocated_reverse_input_ptr_);
    ReverseSequenceData(allocated_reverse_input_ptr_.get(), static_cast<const T*>(X.DataRaw()), sequence_lens_,
                        seq_length_, batch_size_, input_size_, /*num_directions*/ 1);
    input_ptr_ = allocated_reverse_input_ptr_.get();
  } else {
    input_ptr_ = static_cast<const T*>(X.DataRaw());
  }

  // initialize states with optional initial_h and initial_c.
  // After initialization, the memory region of [H_in_ptr_, H_out_ptr) will be initialized
  // with the values of initial_h if it's presented. Otherwise, the region will contains all zeros.
  // initial_H is at LSTM.Inputs[5]
  // initial_C is at LSTM.Inputs[6]
  std::vector<int64_t> initializer_indices = {5, 6};
  std::vector<int64_t> initializer_sizes(2, batch_size_ * hidden_size_);
  std::vector<void*> target_state_ptrs = {static_cast<void*>(H_in_ptr_), static_cast<void*>(C_in_ptr_)};
  InitializeInStates(context, initializer_indices, initializer_sizes,
                     target_state_ptrs, size_of_tensor_elem_, GetSequenceOffset());

  // initialize all tvm args
  InitializeTVMArgs(packed_func_ctx);

  return Status::OK();
}

template <typename T>
void DLLSTMState<T>::FillTVMArgs(LoopPackedFuncCtxOld& packed_func_ctx) {
  // update in states
  packed_func_ctx.tvm_tensors[0].data = static_cast<void*>(H_in_ptr_);
  packed_func_ctx.tvm_tensors[1].data = static_cast<void*>(C_in_ptr_);

  // update input X
  // TODO: we may need to remove this once we move input project to the outside of the sequence loop.
  int X_idx = gsl::narrow_cast<int>(in_state_tensors_.size());
  packed_func_ctx.tvm_tensors[X_idx].data = static_cast<void*>(const_cast<T*>(input_ptr_));

  // update out states
  int out_state_idx = packed_func_ctx.existing_input_count +
                      gsl::narrow_cast<int>(in_state_tensors_.size());
  packed_func_ctx.tvm_tensors[out_state_idx].data = static_cast<void*>(H_out_ptr_);
  packed_func_ctx.tvm_tensors[out_state_idx + 1].data = static_cast<void*>(C_out_ptr_);

#ifdef ENABLE_DUMP_RNN
  std::cout << "FillTVMArgs: curr_loop_step = " << current_loop_step_
            << " (" << LoopDirectionToStr(direction_) << ")\n";
#endif  // ENABLE_DUMP_RNN
  DumpRNNArray("FillTVMArgs: X", input_ptr_, {batch_size_, input_size_});
  DumpRNNArray("FillTVMArgs: H_in", H_in_ptr_, {batch_size_, hidden_size_});
  DumpRNNArray("FillTVMArgs: C_in", C_in_ptr_, {batch_size_, hidden_size_});
  // we don't need to update outputs, so we are done
}

template <typename T>
void DLLSTMState<T>::LoopFinalize() {
  // copy last outpout to Y_h
  if (Y_ptr_ && Y_h_ptr_) {
    for (int i = 0; i < batch_size_; ++i) {
      auto seq_len = sequence_lens_[i];
      auto output_offset = (seq_len - 1) * output_step_stride_ + i * hidden_size_;
      auto Y_H_offset = i * hidden_size_;
      memcpy(Y_h_ptr_ + Y_H_offset, Y_ptr_ + output_offset, hidden_size_ * size_of_tensor_elem_);
    }
  }

  if (Y_ptr_ && direction_ == LoopDirection::kReverse) {
    Tensor* output_tensor = context_->Output(0, TensorShape({seq_length_, num_directions_, batch_size_, hidden_size_}));
    T* orig_Y_ptr = static_cast<T*>(output_tensor->MutableDataRaw());
    DumpRNNArray("LoopFinalize: before reversing orig_Y",
                 orig_Y_ptr, {seq_length_, num_directions_, batch_size_, hidden_size_});
    if (IsSecondReverseLoop()) {
      // For bidirectional LSTM, reverse Y_ptr has shape of {seq_length_, batch_size_, hidden_size_}
      DumpRNNArray("LoopFinalize: before reversing Y",
                   Y_ptr_, {seq_length_, batch_size_, hidden_size_});
    } else {
      DumpRNNArray("LoopFinalize: before reversing Y",
                   Y_ptr_, {seq_length_, num_directions_, batch_size_, hidden_size_});
    }

    // For bidirectional LSTM, we need to adjust the dest pointer for reverse loop
    ReverseSequenceData(orig_Y_ptr + GetSequenceOffset(), Y_ptr_, sequence_lens_,
                        seq_length_, batch_size_, hidden_size_, num_directions_);
    DumpRNNArray("LoopFinalize: after reversing Y",
                 orig_Y_ptr, {seq_length_, num_directions_, batch_size_, hidden_size_});
  } else if (Y_ptr_) {
    DumpRNNArray("LoopFinalize: Y", Y_ptr_, {seq_length_, num_directions_, batch_size_, hidden_size_});
  }

  if (Y_h_ptr_) {
    // move Y_h_ptr to it's head
    DumpRNNArray("LoopFinalize: Y_h", Y_h_ptr_ - GetSequenceOffset(), {num_directions_, batch_size_, hidden_size_});
  }

  if (Y_c_ptr_) {
    // move Y_c_ptr to it's head
    DumpRNNArray("LoopFinalize: Y_c", Y_c_ptr_ - GetSequenceOffset(), {num_directions_, batch_size_, hidden_size_});
  }
}

template class DLLSTMState<float>;
template class DLLSTMState<double>;

#endif

}  // namespace tvm_codegen
}  // namespace onnxruntime
