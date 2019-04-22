// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop_type/scan_state.h"

#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/subgraph_gen_stats.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace tvm_codegen {

static void GetSlicedShapeAndSize(const Tensor* tensor,
                                  int start,
                                  std::vector<int64_t>& sliced_shape,
                                  size_t& total_size) {
  total_size = tensor->DataType()->Size();
  const auto& shape = tensor->Shape().GetDims();
  int dim = gsl::narrow_cast<int>(shape.size());
  ORT_ENFORCE_DEBUG(start < dim);
  sliced_shape.resize(dim - start);
  for (int i = start; i < dim; ++i) {
    int64_t d = shape[i];
    sliced_shape[i - start] = d;
    total_size *= d;
  }
}

void DLScanState::Advance() {
  ORT_ENFORCE_DEBUG(current_loop_step_ < max_loop_step_);

  // TODO: support batch > 1 with different sizes of sequences

  // update inputs
  for (int i = 0; i < input_ptrs_.size(); i++) {
    input_ptrs_[i] = (static_cast<char*>(input_ptrs_[i]) + input_strides_[i]);
  }

  // update outputs
  for (int i = 0; i < output_ptrs_.size(); i++) {
    output_ptrs_[i] = (static_cast<char*>(output_ptrs_[i]) + output_strides_[i]);
  }

  // update input and output states
  if (current_loop_step_ == 0) {
    // When executed the first loop (current_loop_step == 0),
    // assign current_state_input as current_state_output
    // and current_state_output as state_output_buffers_
    for (int i = 0; i < num_loop_state_variables_; i++) {
      current_state_input_ptrs_[i] = current_state_output_ptrs_[i];

      int index = state_to_output_index_[i];
      if (index >= 0) {
        current_state_output_ptrs_[i] = output_ptrs_[index];
      } else {
        current_state_output_ptrs_[i] = state_output_buffers_[i];
      }
    }
  } else if (current_loop_step_ == (max_loop_step_ - 1)) {
    // When executed the last loop step
    // copy from current_state_output_ptrs to state_output_ptrs if needed
    for (int i = 0; i < num_loop_state_variables_; i++) {
      if (state_output_ptrs_[i] && current_state_output_ptrs_[i] != state_output_ptrs_[i])
        memcpy(state_output_ptrs_[i], current_state_output_ptrs_[i], state_bytes_size_[i]);
    }
  } else {
    // When current_loop_step > 0
    // Swap current_state_input_ptrs_[i] and current_state_output_ptrs_[i]
    for (int i = 0; i < num_loop_state_variables_; i++) {
      int index = state_to_output_index_[i];
      if (index >= 0) {
        current_state_input_ptrs_[i] = current_state_output_ptrs_[i];
        current_state_output_ptrs_[i] = output_ptrs_[index];
      } else {
        std::swap(current_state_input_ptrs_[i], current_state_output_ptrs_[i]);
      }
    }
  }

  // increase loop index
  ++current_loop_step_;
}

void DLScanState::InitContext(OpKernelContext* ctx_kernel, PackedFuncCallCtx& ctx_call) {
  ProtoHelperNodeContext ctx(node_);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  // set num_scan_inputs_ as number of input
  bool attr_is_ok = attrs.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs_).IsOK();
  ORT_UNUSED_PARAMETER(attr_is_ok);
  ORT_ENFORCE_DEBUG(attr_is_ok);

  if (!attrs.GetAttrs("axes", input_sequence_axes_).IsOK()) {
    input_sequence_axes_.resize(num_scan_inputs_, 0);
  } else {
    ORT_NOT_IMPLEMENTED("Not implemented for non default axes attribute");
  }

  num_variadic_inputs_ = ctx_kernel->NumVariadicInputs(0);
  num_variadic_outputs_ = ctx_kernel->OutputCount();
  num_loop_state_variables_ = num_variadic_inputs_ - gsl::narrow_cast<int>(num_scan_inputs_);

  if (attrs.GetAttrs<int64_t>("scan_input_directions", input_directions_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(input_directions_.size()) == num_scan_inputs_,
                "Number of entries in 'directions' was ", input_directions_.size(),
                ". Must match 'num_scan_inputs' of ", num_scan_inputs_);
    ORT_ENFORCE(std::all_of(input_directions_.cbegin(), input_directions_.cend(),
                            [](int64_t i) { return i == static_cast<int64_t>(LoopDirection::kForward) ||
                                                   i == static_cast<int64_t>(LoopDirection::kReverse); }),
                "Invalid values in 'scan_input_directions'. 0 == forward. 1 == reverse.");
  } else {
    // default to forward
    input_directions_ = std::vector<int64_t>(num_scan_inputs_, static_cast<int64_t>(LoopDirection::kForward));
  }

  auto num_scan_outputs = num_variadic_outputs_ - num_loop_state_variables_;
  if (attrs.GetAttrs<int64_t>("scan_output_directions", output_directions_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(output_directions_.size()) == num_scan_outputs,
                "Number of entries in 'directions' was ", output_directions_.size(),
                ". Must match 'num_scan_outputs' of ", num_scan_outputs);
    ORT_ENFORCE(std::all_of(output_directions_.cbegin(), output_directions_.cend(),
                            [](int64_t i) { return i == static_cast<int64_t>(LoopDirection::kForward) ||
                                                   i == static_cast<int64_t>(LoopDirection::kReverse); }),
                "Invalid values in 'scan_output_directions'. 0 == forward. 1 == reverse.");
  } else {
    // default to forward
    output_directions_ = std::vector<int64_t>(num_scan_outputs, static_cast<int64_t>(LoopDirection::kForward));
  }

  // TODO: remove alias out of runtime
  // Construct alias_def lookup table from key to state id
  std::map<NodeKey, int> visited_state_alias_def;
  auto subgraph = GetSubgraph(node_);
  for (int i = 0; i < num_loop_state_variables_; i++) {
    auto def = subgraph->GetOutputs()[i];
    auto input_def = codegen::SubGraphStats::Promote(ctx_codegen_.GetGraphStats())->SourceDefOfOutputAlias(def);

    state_to_output_index_.push_back(-1);

    if (input_def) {
      auto key = GetKey(input_def);
      if (visited_state_alias_def.count(key) == 0) {
        visited_state_alias_def.insert(std::make_pair(key, i));
        is_state_alias_.push_back(true);
        continue;
      }
    }
    is_state_alias_.push_back(false);
  }

  int out_idx = 0;
  for (int i = num_loop_state_variables_; i < num_variadic_outputs_; ++i) {
    auto def = subgraph->GetOutputs()[i];

    auto input_def = codegen::SubGraphStats::Promote(ctx_codegen_.GetGraphStats())->SourceDefOfOutputAlias(def);
    if (input_def) {
      auto key = GetKey(input_def);
      if (visited_state_alias_def.count(key) > 0) {
        is_output_alias_.push_back(true);
        auto index = visited_state_alias_def[key];

        state_to_output_index_[index] = out_idx;
        output_to_state_index_.push_back(index);
        out_idx++;
        continue;
      }
    }
    output_to_state_index_.push_back(-1);
    is_output_alias_.push_back(true);
    out_idx++;
  }

  int tvm_arg_idx = 0;

  // Initialize Scan's state inputs and assign them to tvm args
  for (int i = 0; i < num_loop_state_variables_; ++i) {
    const Tensor* t = ctx_kernel->Input<Tensor>(i);

    // set current state_input as state_input
    current_state_input_ptrs_.push_back(const_cast<void*>(t->DataRaw()));

    // assign it to tvm arg
    std::vector<int64_t> dl_shape;
    size_t total_size = 0;
    GetSlicedShapeAndSize(t, /*start*/ 0, dl_shape, total_size);
    DLDataType dl_type = ToTvmDLDataType(t->DataType());
    InitializeOneTVMArg(ctx_call, tvm_arg_idx, dl_shape, dl_type);

    // Also alloc State buffers here
    allocated_input_state_buffers_.push_back(ctx_codegen_.Allocate(total_size));
    allocated_output_state_buffers_.push_back(ctx_codegen_.Allocate(total_size));
    state_input_buffers_.push_back(allocated_input_state_buffers_.back().get());
    state_output_buffers_.push_back(allocated_output_state_buffers_.back().get());

    state_bytes_size_.push_back(total_size);
  }

  // Initialize Scan's inputs and assign them to tvm args
  for (int i = num_loop_state_variables_; i < num_variadic_inputs_; ++i) {
    const Tensor* t = ctx_kernel->Input<Tensor>(i);

    // check whether it is a initializer
    const NodeArg* def = node_.InputDefs()[i];
    bool is_initializer = ctx_codegen_.IsInitializerMarshalled(def->Name());
    input_is_initializer_.push_back(is_initializer);
    if (is_initializer) {
      // if so, continue
      continue;
    }

    // assign it to tvm arg
    std::vector<int64_t> dl_shape;
    size_t total_size = 0;
    GetSlicedShapeAndSize(t, /*start*/ 1, dl_shape, total_size);
    DLDataType dl_type = ToTvmDLDataType(t->DataType());
    InitializeOneTVMArg(ctx_call, tvm_arg_idx, dl_shape, dl_type);

    // update max_sequence length
    auto shape = t->Shape();
    if (max_seq_length_ == 0) {
      max_seq_length_ = shape[0];
    } else {
      ORT_ENFORCE(max_seq_length_ == shape[0]);
    }

    // assing strides and input ptr
    auto in_idx = i - num_loop_state_variables_;
    bool is_forward = (input_directions_[in_idx] == static_cast<int>(LoopDirection::kForward));
    if (is_forward) {
      input_ptrs_.push_back(const_cast<void*>(t->DataRaw()));
      input_strides_.push_back(gsl::narrow_cast<int64_t>(total_size));
    } else {
      input_ptrs_.push_back(const_cast<uint8_t*>(static_cast<const uint8_t*>(t->DataRaw()) + total_size * (max_seq_length_ - 1)));
      input_strides_.push_back(-gsl::narrow_cast<int64_t>(total_size));
    }
  }

  // for now
  min_loop_step_ = 0;
  max_loop_step_ = max_seq_length_;

  // Assign all subgraph's initializiers into tvm args
  for (const auto& item : ctx_codegen_.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      bool is_marshalled = info.layout_info->is_marshalled;
      const Tensor* input_tensor;
      if (is_marshalled) {
        input_tensor = info.layout_info->marshalled_initializer.get();
      } else {
        input_tensor = info.original_initializer;
      }
      // assign it to tvm arg
      std::vector<int64_t> dl_shape = input_tensor->Shape().GetDims();
      DLDataType dl_type = ToTvmDLDataType(input_tensor->DataType());
      InitializeOneTVMArg(ctx_call, tvm_arg_idx, dl_shape, dl_type, input_tensor->DataRaw());
    }
  }

  // Initialize Scan's state output and assign them to tvm args
  for (int i = 0; i < num_loop_state_variables_; ++i) {
    // shape calculate, note loop_state_output does not have sequence dim
    auto shape_proto = node_.OutputDefs()[i]->Shape();
    // output_dims is [batch, rest...]
    std::vector<int64_t> output_dims;

    for (int d = 0; d < shape_proto->dim_size(); ++d) {
      ORT_ENFORCE_DEBUG(shape_proto->dim(d).has_dim_value(), "Unknown static shapes");
      if (shape_proto->dim(d).has_dim_value()) {
        output_dims.push_back(shape_proto->dim(d).dim_value());
      }
    }
    output_dims_.push_back(output_dims);

    // bind it to state_output_ptrs_
    Tensor* t = ctx_kernel->Output(i, TensorShape(output_dims));
    state_output_ptrs_.push_back(t->MutableDataRaw());

    // temporarily set current state_output as state_input_buffers
    current_state_output_ptrs_.push_back(state_input_buffers_[i]);

    // assign it to tvm arg
    std::vector<int64_t> dl_shape = t->Shape().GetDims();
    DLDataType dl_type = ToTvmDLDataType(t->DataType());
    InitializeOneTVMArg(ctx_call, tvm_arg_idx, dl_shape, dl_type);
  }

  // Initialize Scan's output
  for (int i = num_loop_state_variables_; i < num_variadic_outputs_; ++i) {
    auto out_idx = i - num_loop_state_variables_;
    // handle shape
    auto shape_proto = node_.OutputDefs()[i]->Shape();

    // output_dims is [seq, rest...]
    std::vector<int64_t> output_dims = {max_seq_length_};

    for (int d = 1; d < shape_proto->dim_size(); ++d) {
      auto dim_elem = shape_proto->dim(d);
      ORT_ENFORCE_DEBUG(dim_elem.has_dim_value(), "Unknown static shapes");
      output_dims.push_back(dim_elem.dim_value());
    }
    output_dims_.push_back(output_dims);

    Tensor* t = ctx_kernel->Output(i, TensorShape(output_dims));

    size_t total_size = 0;
    std::vector<int64_t> dl_shape;
    GetSlicedShapeAndSize(t, /*start*/ 1, dl_shape, total_size);

    // assing strides and output ptr
    bool is_forward = (output_directions_[i - num_loop_state_variables_] == static_cast<int>(LoopDirection::kForward));
    if (is_forward) {
      output_ptrs_.push_back(t->MutableDataRaw());
      output_strides_.push_back(gsl::narrow_cast<int64_t>(total_size));
    } else {
      output_ptrs_.push_back(static_cast<uint8_t*>(t->MutableDataRaw()) + total_size * (max_seq_length_ - 1));
      output_strides_.push_back(-gsl::narrow_cast<int64_t>(total_size));
    }

    if (!is_output_alias_[out_idx]) {
      DLDataType dl_type = ToTvmDLDataType(t->DataType());
      InitializeOneTVMArg(ctx_call, tvm_arg_idx, dl_shape, dl_type);
    } else {
      int index = output_to_state_index_[out_idx];
      // update current_state_output as output
      current_state_output_ptrs_[index] = output_ptrs_.back();
    }
  }
}

// Initialize Loop by assigning inputs/outpus/, with reuse of previous setup
void DLScanState::UpdateContext(OpKernelContext* ctx_kernel, PackedFuncCallCtx& ctx_call) {
  // Assign variables of ONNX
  num_variadic_inputs_ = ctx_kernel->NumVariadicInputs(0);
  num_variadic_outputs_ = ctx_kernel->OutputCount();
  num_loop_state_variables_ = num_variadic_inputs_ - gsl::narrow_cast<int>(num_scan_inputs_);

  // Initialize Scan's state inputs and assign them to tvm args
  ORT_ENFORCE_DEBUG(num_loop_state_variables_ == current_state_input_ptrs_.size());
  for (int i = 0; i < num_loop_state_variables_; ++i) {
    const Tensor* t = ctx_kernel->Input<Tensor>(i);

    // set current state_input as state_input
    current_state_input_ptrs_[i] = const_cast<void*>(t->DataRaw());

    // reset state_input/outpu_buffers
    state_input_buffers_[i] = allocated_input_state_buffers_[i].get();
    state_output_buffers_[i] = allocated_output_state_buffers_[i].get();
  }

  // Initialize Scan's inputs and assign them to tvm args
  for (int i = num_loop_state_variables_; i < num_variadic_inputs_; ++i) {
    const Tensor* t = ctx_kernel->Input<Tensor>(i);

    // check whether it is a initializer
    if (input_is_initializer_[i - num_loop_state_variables_]) {
      // if so, continue
      continue;
    }

    // update max_sequence length
    const auto& shape = t->Shape();

    // TODO: handle dynamic batch size too
    if (max_seq_length_ == 0) {
      max_seq_length_ = shape[0];
    } else {
      ORT_ENFORCE_DEBUG(max_seq_length_ == shape[0]);
    }

    // assign to input_ptrs;
    if (input_directions_[i - num_loop_state_variables_] == static_cast<int64_t>(LoopDirection::kForward)) {
      input_ptrs_[i - num_loop_state_variables_] = const_cast<void*>(t->DataRaw());
    } else {
      input_ptrs_[i - num_loop_state_variables_] = static_cast<uint8_t*>(const_cast<void*>(t->DataRaw())) - input_strides_[i - num_loop_state_variables_] * (max_seq_length_ - 1);
    }
  }

  // for now
  min_loop_step_ = 0;
  max_loop_step_ = max_seq_length_;

  // Initialize Scan's state output and assign them to tvm args
  for (int i = 0; i < num_loop_state_variables_; ++i) {
    // shape calculate
    // TODO: handle dynamic batch_size
    std::vector<int64_t>& output_dims = output_dims_[i];

    // bind it to state_output_ptrs_
    Tensor* t = ctx_kernel->Output(i, TensorShape::ReinterpretBaseType(output_dims));
    state_output_ptrs_[i] = t->MutableDataRaw();

    // temporarily set current state_output as state_input_buffers
    current_state_output_ptrs_[i] = state_input_buffers_[i];
  }

  // Initialize Scan's output
  for (int i = num_loop_state_variables_; i < num_variadic_outputs_; ++i) {
    // shape calculate
    // TODO: handle dynamic batch_size
    auto& output_dims = output_dims_[i];
    output_dims[0] = max_seq_length_;

    Tensor* t = ctx_kernel->Output(i, TensorShape::ReinterpretBaseType(output_dims));
    int out_idx = i - num_loop_state_variables_;
    if (output_directions_[out_idx] == static_cast<int64_t>(LoopDirection::kForward)) {
      output_ptrs_[out_idx] = t->MutableDataRaw();
    } else {
      output_ptrs_[out_idx] = static_cast<uint8_t*>(t->MutableDataRaw()) - output_strides_[out_idx] * (max_seq_length_ - 1);
    }

    if (is_output_alias_[out_idx]) {
      int index = output_to_state_index_[out_idx];
      // update current_state_output as output
      current_state_output_ptrs_[index] = output_ptrs_[i - num_loop_state_variables_];
    }
  }

  // reset loop count
  current_loop_step_ = 0;
}

void DLScanState::FillTVMArgs(PackedFuncCallCtx& ctx_call) {
  size_t arg_index = 0;

  // update input states
  for (auto i : current_state_input_ptrs_) {
    ctx_call.tvm_tensors[arg_index].data = i;
    ++arg_index;
  }

  // update inputs
  for (auto i : input_ptrs_) {
    // i could nullptr, when it is an subgraph's initializer
    if (i) {
      ctx_call.tvm_tensors[arg_index].data = i;
      ++arg_index;
    }
  }

  // shift arg_index by initalizer size since intializers don't need to update
  arg_index += ctx_codegen_.SizeInitializerMarshalled();

  // update output states
  for (auto& i : current_state_output_ptrs_) {
    ctx_call.tvm_tensors[arg_index].data = i;
    ++arg_index;
  }

  // update outputs
  for (int i = 0; i < output_ptrs_.size(); ++i) {
    if (!is_output_alias_[i]) {
      ctx_call.tvm_tensors[arg_index].data = output_ptrs_[i];
      ++arg_index;
    }
  }
}

void DLScanState::LoopFinalize() {
  // TODO: Add a flag to enable or disable during runtime for truncated sequence
  max_seq_length_ = 0;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
