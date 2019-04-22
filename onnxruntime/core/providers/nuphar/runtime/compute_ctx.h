// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/func_api.h"
#include "core/framework/func_kernel.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "gsl/gsl_util"

namespace onnxruntime {
namespace tvm_codegen {

using DataAllocFunc = std::function<void*(size_t)>;

// this class contains ORT compute context from NodeComputeInfo
class NupharComputeCtx {
 public:
  explicit NupharComputeCtx(
      DataAllocFunc data_alloc_func)
      : data_alloc_func_(data_alloc_func),
        input_tensors_(nullptr),
        num_inputs_(0),
        output_tensors_(nullptr),
        num_outputs_(0),
        op_kernel_ctx_(nullptr) {
  }

  void Bind(
      ONNXRunTimeTensor* input_tensors,
      size_t num_inputs,
      ONNXRunTimeTensor* output_tensors,
      size_t num_outputs) {
    input_tensors_ = input_tensors;
    num_inputs_ = num_inputs;
    output_tensors_ = output_tensors;
    num_outputs_ = num_outputs;
    op_kernel_ctx_ = nullptr;
    output_dtypes_.resize(num_outputs);
  }

  void Bind(OpKernelContext* op_kernel_ctx) {
    op_kernel_ctx_ = op_kernel_ctx;
  }

  inline const void* InputData(int index) const {
    if (op_kernel_ctx_) {
      return op_kernel_ctx_->Input<Tensor>(index)->DataRaw();
    } else {
      const auto& t = input_tensors_[index];
      return t.data;
    }
  }

  inline const int64_t* InputShape(int index, size_t& shape_rank) const {
    if (op_kernel_ctx_) {
      const auto& s = op_kernel_ctx_->Input<Tensor>(index)->Shape();
      shape_rank = s.NumDimensions();
      return s.GetDims().data();
    } else {
      ORT_ENFORCE_DEBUG(index < num_inputs_);
      const auto& t = input_tensors_[index];
      shape_rank = t.ndim;
      return t.shape;
    }
  }

  inline std::tuple<const void*, const int64_t*, size_t, MLDataType> Input(int index) const {
    if (op_kernel_ctx_) {
      const auto& t = op_kernel_ctx_->Input<Tensor>(index);
      return std::make_tuple(
          t->DataRaw(),
          t->Shape().GetDims().data(),
          t->Shape().NumDimensions(),
          t->DataType());
    } else {
      ORT_ENFORCE_DEBUG(index < num_inputs_);
      const auto& t = input_tensors_[index];
      return std::make_tuple(
          t.data,
          t.shape,
          t.ndim,
          ToMLDataType(t.dtype));
    }
  }

  inline void* Output(int index, const TensorShape& shape, MLDataType dtype) {
    if (op_kernel_ctx_ != nullptr) {
      auto t = op_kernel_ctx_->Output(index, shape);
      ORT_ENFORCE_DEBUG(dtype == t->DataType());
      return t->MutableDataRaw();
    } else {
      ORT_ENFORCE_DEBUG(index < num_outputs_);
      auto& t = output_tensors_[index];
      output_dtypes_[index] = dtype;
      t.data = data_alloc_func_(shape.Size() * dtype->Size());
      t.ndim = shape.NumDimensions();
      t.dtype = ORT_type_to_c_type(dtype);
      t.shape = new int64_t[t.ndim];  // NOTE: this is freed in func_kernel.h, line 73
      memcpy(t.shape, shape.GetDims().data(), t.ndim * sizeof(int64_t));
      return t.data;
    }
  }

  inline void* Output(int index, const TensorShape& shape) {
    ORT_ENFORCE_DEBUG(op_kernel_ctx_ != nullptr);
    auto t = op_kernel_ctx_->Output(index, shape);
    return t->MutableDataRaw();
  }

  inline void* OutputWithKnownDType(int index, const TensorShape& shape) {
    if (op_kernel_ctx_ != nullptr) {
      return Output(index, shape);
    } else {
      const auto& dtype = output_dtypes_[index];
      return Output(index, shape, dtype);
    }
  }

  inline int InputCount() const {
    if (op_kernel_ctx_)
      return op_kernel_ctx_->InputCount();
    else
      return gsl::narrow_cast<int>(num_inputs_);
  }

  inline int OutputCount() const {
    if (op_kernel_ctx_)
      return op_kernel_ctx_->OutputCount();
    else
      return gsl::narrow_cast<int>(num_outputs_);
  }

  inline OpKernelContext* GetOpKernelContext() const {
    return op_kernel_ctx_;
  }

 private:
  static MLDataType ToMLDataType(DType t) {
    switch (t) {
      case DType::TFloat32:
        return DataTypeImpl::GetType<float>();
      case DType::TInt32:
        return DataTypeImpl::GetType<int32_t>();
      case DType::TDouble:
        return DataTypeImpl::GetType<double>();
      case DType::TInt64:
        return DataTypeImpl::GetType<int64_t>();
      case DType::TInt8:
        return DataTypeImpl::GetType<int8_t>();
      case DType::TUint8:
        return DataTypeImpl::GetType<uint8_t>();
      case DType::TInt16:
        return DataTypeImpl::GetType<int16_t>();
    }
    ORT_NOT_IMPLEMENTED("Unsupported DType");
  }

  DataAllocFunc data_alloc_func_;
  ONNXRunTimeTensor* input_tensors_;
  size_t num_inputs_;
  ONNXRunTimeTensor* output_tensors_;
  size_t num_outputs_;
  std::vector<MLDataType> output_dtypes_;

  OpKernelContext* op_kernel_ctx_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
