// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Add(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name = "add");
tvm::Tensor Add(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name = "add");
tvm::Tensor Add(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name = "add");
tvm::Tensor Div(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name = "div");
tvm::Tensor Div(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name = "div");
tvm::Tensor Div(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name = "div");
tvm::Tensor Equal(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name = "equal");
tvm::Tensor Equal(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name = "equal");
tvm::Tensor Equal(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name = "equal");
tvm::Tensor Mul(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name = "mul");
tvm::Tensor Mul(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name = "mul");
tvm::Tensor Mul(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name = "mul");
tvm::Tensor PRelu(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name = "prelu");
tvm::Tensor PRelu(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name = "prelu");
tvm::Tensor Sub(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name = "sub");
tvm::Tensor Sub(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name = "sub");
tvm::Tensor Sub(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name = "sub");

}  // namespace tvm_codegen
}  // namespace onnxruntime
