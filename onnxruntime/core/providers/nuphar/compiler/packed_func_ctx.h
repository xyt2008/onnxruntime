// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tvm/build_module.h>

#pragma once
namespace onnxruntime {
namespace tvm_codegen {

// PackedFuncInfo holds tvm::runtime::PackedFunc (the generated function)
// And corresponding meta information to call it, like number of argument and offset
// The owner of PackedFuncInfo will be OpKernel.
// But it is created in TVMCompiler.
// It will be consumed in TVMExecutor.
// Note: PackedFuncInfo only contains meta data from code generation
///      not including runtime information.
struct PackedFuncInfo {
  std::string name;
  int num_args;

  std::vector<int> type_codes;
  tvm::runtime::PackedFunc packed_func;
  DLDeviceType device_type;

  // meta data for decoupling compiler and runtime
  // TODO confirm this
  size_t input_count;
  size_t output_count;
  size_t num_actual_outputs;

  // TODO: refactor these two after alias
  std::vector<int> output_tvm_arg_idx;
  std::vector<int> output_def_alias;

  //TODO: remove tvm_outputs after usage tvm_outputs in ExecBlock
  // Why there is tvm_outputs
  tvm::Array<tvm::Tensor> tvm_outputs;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
