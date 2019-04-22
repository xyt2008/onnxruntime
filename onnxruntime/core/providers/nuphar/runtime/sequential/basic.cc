// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/basic.h"

#include "core/codegen/common/common.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "core/codegen/target/tvm_context.h"
#include "gsl/gsl_util"
#include <tvm/tvm.h>

// from onnxruntime_typeinf.cc, in global namespace
const onnxruntime::DataTypeImpl* ElementTypeFromProto(int type);

namespace onnxruntime {
namespace tvm_codegen {

// TODO: remove node, after spliting shape update and pointer update
// TODO: remove CodeGenContext after splitting codegen and runtime
void BasicExecBlock::Run(NupharComputeCtx* compute_ctx,
                         PackedFuncCallCtx* ctx_call,
                         const Node& node,
                         const NupharCodeGenCtx& ctx_codegen) {
  if (ctx_call->HasInitialized()) {
    UpdateContext(compute_ctx, ctx_call, node, ctx_codegen);
  } else {
    InitContext(compute_ctx, ctx_call, node, ctx_codegen);
  }

  const PackedFuncInfo& info = ctx_call->info;
  auto& output_tvm_arg_idx = info.output_def_alias;
  auto& output_def_alias = info.output_def_alias;

  tvm::TVMArgs tvm_args(ctx_call->lvalues.data(),
                        info.type_codes.data(),
                        info.num_args);

  size_t num_actual_outputs = info.num_actual_outputs;

  tvm::TVMRetValue rvalue;
  const tvm::runtime::PackedFunc& func = info.packed_func;
  func.CallPacked(tvm_args, &rvalue);

  if (num_actual_outputs < compute_ctx->OutputCount()) {
    // some outputs are aliased, need to copy
    for (int i_def = 0; i_def < compute_ctx->OutputCount(); ++i_def) {
      if (output_def_alias[i_def] == i_def)
        continue;
      int tvm_arg_idx = output_tvm_arg_idx[output_def_alias[i_def]];
      const auto& shape = TensorShape::ReinterpretBaseType(ctx_call->shapes[tvm_arg_idx]);
      MLDataType src_dtype = ElementTypeFromProto(node.OutputDefs()[output_def_alias[i_def]]->TypeAsProto()->tensor_type().elem_type());
      MLDataType dst_dtype = ElementTypeFromProto(node.OutputDefs()[i_def]->TypeAsProto()->tensor_type().elem_type());
      void* src = compute_ctx->Output(output_def_alias[i_def], shape, src_dtype);
      void* dst = compute_ctx->Output(i_def, shape, dst_dtype);
      // TODO: change it to use provider::CopyTensor for non-CPU devices
      memcpy(dst, src, shape.Size());
    }
  }
}

// TODO: please refactor these
// TODO: ask Ke to split pointer update and shape update
// TODO: remove node, after spliting shape update and pointer update
// TODO: remove CodeGenContext after splitting codegen and runtime
void BasicExecBlock::InitContext(NupharComputeCtx* compute_ctx,
                                 PackedFuncCallCtx* ctx_call,
                                 const Node& node,
                                 const NupharCodeGenCtx& ctx_codegen) {
  const auto& tvm_ctx = ctx_call->tvm_ctx;
  const PackedFuncInfo& info = ctx_call->info;

  // TODO: don't account for initializer in Reshape

  size_t input_count = info.input_count;
  size_t output_count = info.output_count;
  size_t num_actual_outputs = info.num_actual_outputs;
  auto& output_tvm_arg_idx = info.output_tvm_arg_idx;
  auto& output_def_alias = info.output_def_alias;

  size_t num_args = input_count + num_actual_outputs;
  auto& lvalues = ctx_call->lvalues;
  lvalues.resize(num_args);
  auto& tvm_tensors = ctx_call->tvm_tensors;
  tvm_tensors.resize(num_args);
  auto& shapes = ctx_call->shapes;
  shapes.resize(num_args);

  auto& realized_dims = ctx_call->realized_dims;

  auto fill_input = [&](size_t i, const void* input_data, const int64_t* input_shape, size_t shape_rank, MLDataType data_type) {
    shapes[i] = std::vector<int64_t>(input_shape, input_shape + shape_rank);
    ORT_ENFORCE_DEBUG(ctx_codegen.GetCodeGenHandle()->allow_unaligned_buffers || (reinterpret_cast<std::uintptr_t>(input_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);
    tvm_tensors[i] = {const_cast<void*>(input_data), tvm_ctx,
                      gsl::narrow_cast<int>(shapes[i].size()), dtype,
                      shapes[i].data(), nullptr, 0};
    lvalues[i].v_handle = &(tvm_tensors[i]);
  };

  // non-initializer inputs
  auto& context_idx_to_tvm_tensors = ctx_call->context_idx_to_tvm_tensors;
  context_idx_to_tvm_tensors.resize(compute_ctx->InputCount() + compute_ctx->OutputCount(), nullptr);
  size_t i_input = 0;

  node.ForEachWithIndex(
      node.InputDefs(),
      [&](const NodeArg& def, size_t i_def) {
        // TODO remove usage of ctx_codegen
        if (ctx_codegen.IsInitializerMarshalled(def.Name())) {
          return Status::OK();  // skip initializers
        }

        // update pointer
        auto tuple = compute_ctx->Input(i_def);
        const void* input_data = std::get<0>(tuple);
        const int64_t* input_shape = std::get<1>(tuple);
        size_t input_shape_rank = std::get<2>(tuple);
        MLDataType data_type = std::get<3>(tuple);
        context_idx_to_tvm_tensors[i_def] = &(tvm_tensors[i_input]);
        fill_input(i_input, input_data, input_shape, input_shape_rank, data_type);

        // update dynamic shape in realized_dims
        // TODO remove usage of ctx_codegen
        tvm::Tensor tvm_tensor = ctx_codegen.GetTVMTensorCtx().Lookup(&def);
        // const auto& shape = tvm_tensor->nominal_shape;
        int64_t shape_rank = ShapeRank(&def);
        ORT_RETURN_IF_NOT(shape_rank == tvm_tensor->shape.size());
        for (size_t dim = 0; dim < gsl::narrow<size_t>(shape_rank); ++dim) {
          if (!ShapeHasValue(&def, dim)) {
            const std::string& dim_param = ShapeSymbol(&def, dim);
            auto dim_value_iter = realized_dims.find(dim_param);
            if (dim_value_iter == realized_dims.end())
              realized_dims.insert(std::make_pair(dim_param, input_shape[dim]));
            else if (dim_value_iter->second == -1)
              dim_value_iter->second = input_shape[dim];
            else
              ORT_RETURN_IF_NOT(dim_value_iter->second == input_shape[dim]);

            ctx_call->CreateOrUpdateDimPatch<true>(dim_param, i_def, i_input, dim);
          }
        }
        ++i_input;
        return Status::OK();
      });

  // Handle initializers
  // TODO remove usage of ctx_codegen
  for (const auto& item : ctx_codegen.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      bool is_marshalled = info.layout_info->is_marshalled;
      const Tensor* t =
          is_marshalled ? info.layout_info->marshalled_initializer.get()
                        : info.original_initializer;
      fill_input(i_input++, t->DataRaw(), t->Shape().GetDims().data(),
                 t->Shape().NumDimensions(), t->DataType());
    }
  }
  ORT_ENFORCE_DEBUG(i_input == input_count);

  // outputs
  for (size_t i_def = 0; i_def < output_count; ++i_def) {
    if (output_def_alias[i_def] != i_def)
      continue;
    int tvm_arg_idx = output_tvm_arg_idx[i_def];
    ORT_ENFORCE_DEBUG(tvm_arg_idx != OutputAliased);

    // TODO: remove usage of tvm_outputs
    const tvm::Tensor& output = info.tvm_outputs[tvm_arg_idx - input_count];
    const auto& output_shape = output->shape;
    auto output_rank = output_shape.size();
    auto& realized_shape = shapes[tvm_arg_idx];
    realized_shape.resize(output_rank);
    for (size_t index = 0; index < output_rank; ++index) {
      const auto& expr = output_shape[index];
      const int64_t* p = tvm::as_const_int(expr);
      if (nullptr != p) {
        realized_shape[index] = *p;
      } else {
        auto var = expr.as<tvm::Variable>();
        if (var) {
          const auto& var_name = var->name_hint;
          auto dim_value_iter = realized_dims.find(var_name);
          ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());
          // generat output shape
          realized_shape[index] = dim_value_iter->second;
          // update patch
          const std::string& dim_param = var_name;
          ctx_call->CreateOrUpdateDimPatch<false>(dim_param, input_count + i_def, tvm_arg_idx, index);
        }
      }
    }

    MLDataType output_dtype = ElementTypeFromProto(node.OutputDefs()[i_def]->TypeAsProto()->tensor_type().elem_type());
    void* output_data = compute_ctx->Output(i_def, TensorShape(shapes[tvm_arg_idx]), output_dtype);
    ORT_ENFORCE(ctx_codegen.GetCodeGenHandle()->allow_unaligned_buffers || (reinterpret_cast<std::uintptr_t>(output_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(output_dtype);
    tvm_tensors[tvm_arg_idx] = {output_data, tvm_ctx,
                                gsl::narrow_cast<int>(shapes[tvm_arg_idx].size()),
                                dtype, shapes[tvm_arg_idx].data(), nullptr, 0};
    context_idx_to_tvm_tensors[i_def + compute_ctx->InputCount()] = &(tvm_tensors[tvm_arg_idx]);
    lvalues[tvm_arg_idx].v_handle = &(tvm_tensors[tvm_arg_idx]);
  }
}

// TODO: remove node, after spliting shape update and pointer update
void BasicExecBlock::UpdateContext(NupharComputeCtx* compute_ctx,
                                   PackedFuncCallCtx* ctx_call,
                                   const Node& node,
                                   const NupharCodeGenCtx& /*ctx_codegen*/) {
  // existing run_state, only need to fill in non-initializer input/output, and patch dims

  const PackedFuncInfo& info = ctx_call->info;
  auto& output_tvm_arg_idx = info.output_tvm_arg_idx;

  // filling input data pointer
  for (int i = 0; i < compute_ctx->InputCount(); ++i) {
    // skip initializers
    if (ctx_call->context_idx_to_tvm_tensors[i] == nullptr)
      continue;

    DLTensor* tvm_tensor = ctx_call->context_idx_to_tvm_tensors[i];
    tvm_tensor->data = const_cast<void*>(compute_ctx->InputData(i));
  }

  // patch dims
  for (const auto& dim_patch : ctx_call->dim_patches) {
    const auto& context_input_idx = dim_patch.from_context_input_idx_and_dim.first;
    const auto& from_dim = dim_patch.from_context_input_idx_and_dim.second;

    // get actual size of dim
    size_t input_shape_rank;
    const int64_t* input_shape = compute_ctx->InputShape(context_input_idx, input_shape_rank);
    ORT_ENFORCE_DEBUG(from_dim < input_shape_rank);
    int64_t dim_size = input_shape[from_dim];

    // update realized_dims_
    const NodeArg* input_def = node.InputDefs()[context_input_idx];  //TODO remove node usage
    const auto& dim_param = input_def->Shape()->dim(from_dim).dim_param();
    auto dim_value_iter = ctx_call->realized_dims.find(dim_param);
    ORT_ENFORCE_DEBUG(dim_value_iter != ctx_call->realized_dims.end());
    if (dim_value_iter->second == -1)  // -1 means not set in current execution_frame
      dim_value_iter->second = dim_size;
    else
      ORT_ENFORCE(dim_value_iter->second == dim_size);

    // patch DLTensor dims
    for (const auto& p : dim_patch.tvm_arg_idx_and_dim) {
      DLTensor& tvm_tensor = ctx_call->tvm_tensors[p.first];
      tvm_tensor.shape[p.second] = dim_size;
    }
  }

  // Now DLTensor output should have updated dim, process outputs
  for (int i_def = 0; i_def < compute_ctx->OutputCount(); ++i_def) {
    if (output_tvm_arg_idx[i_def] == OutputAliased)
      continue;

    int tvm_arg_idx = output_tvm_arg_idx[i_def];
    DLTensor* tvm_tensor = ctx_call->context_idx_to_tvm_tensors[compute_ctx->InputCount() + i_def];
    void* output_data = compute_ctx->OutputWithKnownDType(i_def, TensorShape::ReinterpretBaseType(ctx_call->shapes[tvm_arg_idx]));
    tvm_tensor->data = output_data;
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
