// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/compute_capability.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"

#ifndef NDEBUG
#define ORT_ENFORCE_DEBUG(...) ORT_ENFORCE(__VA_ARGS__)
#else
#define ORT_ENFORCE_DEBUG(...)
#endif  // !NDEBUG

#define DYNAMIC_PROMOTE_FROM_BASE(X, BASE)                      \
  static inline const X* Promote(const BASE* base) {            \
    auto derived = dynamic_cast<const X*>(base);                \
    ORT_ENFORCE(nullptr != derived);                            \
    return derived;                                             \
  }                                                             \
                                                                \
  static inline X* Promote(BASE* base) {                        \
    auto derived = dynamic_cast<X*>(base);                      \
    ORT_ENFORCE(nullptr != derived);                            \
    return derived;                                             \
  }                                                             \
                                                                \
  static inline X* Promote(const std::unique_ptr<BASE>& base) { \
    auto derived = dynamic_cast<X*>(base.get());                \
    ORT_ENFORCE(nullptr != derived);                            \
    return derived;                                             \
  }                                                             \
                                                                \
  static inline X* Promote(const std::shared_ptr<BASE>& base) { \
    auto derived = dynamic_cast<X*>(base.get());                \
    ORT_ENFORCE(nullptr != derived);                            \
    return derived;                                             \
  }

namespace onnxruntime {

// Nodekey is used as a key for maps
using NodeKey = std::string;

NodeKey GetKey(const onnxruntime::Node* node);
NodeKey GetKey(const onnxruntime::NodeArg* def);

bool IsRecurrentNode(const onnxruntime::Node& node);

// Helper function that creates ComputeCapability for subgraphs
std::unique_ptr<ComputeCapability> ToCapacity(const onnxruntime::GraphViewer& graph,
                                              std::unique_ptr<IndexedSubGraph>& subgraph);

bool IsFusedNode(const Node& node);

bool HasLoop(const Node& node);

const Graph* GetSubgraph(const Node& node);

std::string NormalizeCppName(const std::string& name);

std::string NormalizeNodeArgName(const NodeArg* def);

// Return the corresponding input node for the NodeArg of the given node
const onnxruntime::Node* GetInputNode(const Node& node, const NodeArg* def);

int64_t ShapeRank(const NodeArg* def);

bool ShapeHasValue(const NodeArg* def, int i);

bool ShapeHasSymbol(const NodeArg* def, int i);

int64_t ShapeValue(const NodeArg* def, int i);

const std::string& ShapeSymbol(const NodeArg* def, int i);

ONNX_NAMESPACE::TensorProto_DataType TensorProtoDataType(const NodeArg* def);

}  // namespace onnxruntime
