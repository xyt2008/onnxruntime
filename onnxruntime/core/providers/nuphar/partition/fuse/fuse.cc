// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/nuphar/partition/fuse/fuse.h"
#include "core/providers/nuphar/nuphar_execution_provider.h"
#include "core/providers/nuphar/common/analysis/subgraph_gen_stats.h"
#include "core/providers/nuphar/compiler/tvm_initializer.h"
#include "core/codegen/target/tvm_context.h"

// TODO: remove runtime headers after runtime refactoring
#include "core/providers/nuphar/runtime/sequential/basic.h"
#include "core/providers/nuphar/runtime/sequential/loop.h"

namespace onnxruntime {
namespace nuphar {

thread_local std::unique_ptr<std::unordered_map<const NupharFunctionState*, std::unique_ptr<tvm_codegen::PackedFuncCallCtx>>> NupharFunctionState::tvm_run_state_map_;

// NupharFunctionState is the core of compilation and runtime for Nuphar
// It creates a TVMCompiler, which builds TVM IR and lowers IR to an executable function.
// The flow after refactoring:
// Construct, Build, Lower, ..
// CreateExecutor
//
NupharFunctionState::NupharFunctionState(
    const Node& node,
    TryGetConstantFunc try_get_constant_func,
    const ComputeContext& ctx,
    const NupharExecutionProvider* provider)
    : provider_(provider),
      ctx_(ctx),
      compute_ctx_([this](size_t bytes) {
        return ctx_.allocate_func(ctx_.allocator_handle, 1, bytes);
      }) {
  Init(node, try_get_constant_func);
}

NupharFunctionState::NupharFunctionState(
    const OpKernelInfo& info)
    : provider_(dynamic_cast<const NupharExecutionProvider*>(info.GetExecutionProvider())),
      compute_ctx_(nullptr) {
  Init(info.node(),
       [&](const std::string& name, const Tensor** tensor) {
         return info.TryGetConstantInput(name, tensor);
       });
}

void NupharFunctionState::Init(
    const Node& node,
    TryGetConstantFunc try_get_constant_func) {
  // TODO: rename tvm_target to a proper name
  auto tvm_target = provider_->GetTVMTarget();

  node.ForEachDef(
      [&](const NodeArg& def, bool is_input) {
        if (!is_input)
          return;

        const Tensor* tensor = nullptr;
        if (try_get_constant_func(def.Name(), &tensor)) {
          initializer_map_.emplace(def.Name(), tensor);
        }
      });

  tvm_compiler_ = std::make_unique<tvm_codegen::TVMCompiler>(
      node,
      initializer_map_,
      provider_->GetTVMCodeGenSetting());

  codegen_status_ = tvm_compiler_->Build();

  if (codegen_status_.IsOK()) {
    pack_info_ = std::make_unique<tvm_codegen::PackedFuncInfo>();
    codegen_status_ = tvm_compiler_->Lower(tvm_target,
                                           provider_->GetTVMHostTarget(),
                                           pack_info_.get());
  }

  // TODO: move this to another function
  if (node.OpType() == "Scan") {
    exec_blocks_.push_back(std::move(std::make_unique<tvm_codegen::LoopExecBlock>(tvm_codegen::LoopDirection::kForward,
                                                                                  tvm_codegen::LoopOpKind::kLoopOpScan,
                                                                                  "nuphar_exec_" + node.Name())));
  } else {
    exec_blocks_.push_back(std::move(std::make_unique<tvm_codegen::BasicExecBlock>("nuphar_exec_" + node.Name())));
  }
}

NupharFunctionState::~NupharFunctionState() {
  if (tvm_run_state_map_)
    tvm_run_state_map_->erase(this);
}

Status NupharFunctionState::ComputeInternal() const {
  if (!codegen_status_.IsOK()) {
    return codegen_status_;
  }

  if (nullptr == tvm_run_state_map_) {
    tvm_run_state_map_ = std::make_unique<std::unordered_map<const NupharFunctionState*, std::unique_ptr<tvm_codegen::PackedFuncCallCtx>>>();
  }

  auto& tvm_ctx = provider_->GetTVMContext();
  if (tvm_run_state_map_->find(this) == tvm_run_state_map_->end()) {
    tvm_run_state_map_->emplace(
        std::make_pair(this,
                       std::make_unique<tvm_codegen::PackedFuncCallCtx>(
                           tvm_ctx,
                           *(pack_info_.get()),
                           provider_->GetTLSRealizedDims())));
  }

  tvm_codegen::PackedFuncCallCtx* run_state = tvm_run_state_map_->find(this)->second.get();

  for (auto& exec : exec_blocks_) {
    exec->Run(&compute_ctx_, run_state, tvm_compiler_->GetNode(), tvm_compiler_->GetCodeGenContext());
  }

  return Status::OK();
}

Status NupharFunctionState::Compute(
    ONNXRunTimeTensor* input_tensors,
    size_t num_inputs,
    ONNXRunTimeTensor* output_tensors,
    size_t num_outputs) const {
  compute_ctx_.Bind(input_tensors, num_inputs, output_tensors, num_outputs);
  return ComputeInternal();
}

Status NupharFunctionState::Compute(OpKernelContext* op_kernel_context) const {
  compute_ctx_.Bind(op_kernel_context);
  return ComputeInternal();
}

class NupharKernel : public OpKernel {
 public:
  explicit NupharKernel(const OpKernelInfo& info)
      : OpKernel(info),
        func_state_(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    return func_state_.Compute(context);
  }

 private:
  NupharFunctionState func_state_;
};

}  // namespace nuphar

#define NUPHAR_OP(name, ver, types)                  \
  ONNX_OPERATOR_KERNEL_EX(                           \
      name,                                          \
      kOnnxDomain,                                   \
      ver,                                           \
      kNupharExecutionProvider,                      \
      KernelDefBuilder().TypeConstraint("T", types), \
      nuphar::NupharKernel);

#define NUPHAR_VERSIONED_OP(name, start_ver, end_ver, types) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                         \
      name,                                                  \
      kOnnxDomain,                                           \
      start_ver,                                             \
      end_ver,                                               \
      kNupharExecutionProvider,                              \
      KernelDefBuilder().TypeConstraint("T", types),         \
      nuphar::NupharKernel);

LIST_NUPHAR_OPS()

#undef NUPHAR_OP

// ops that have multiple type constraints

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    6,
    8,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Cast,
    kOnnxDomain,
    9,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kOnnxDomain,
    9,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
    nuphar::NupharKernel);

}  // namespace onnxruntime
