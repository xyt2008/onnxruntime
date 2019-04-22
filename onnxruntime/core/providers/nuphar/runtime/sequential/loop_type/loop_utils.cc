// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop_type/loop_utils.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace tvm_codegen {

LoopOpKind GetLoopOpKind(const Node& node) {
  std::string type = node.OpType();

  if (type == "GRU") {
    return LoopOpKind::kLoopOpGRU;
  } else if (type == "LSTM") {
    return LoopOpKind::kLoopOpLSTM;
  } else if (type == "RNN") {
    return LoopOpKind::kLoopOpRNN;
  } else if (type == "Scan") {
    return LoopOpKind::kLoopOpScan;
  }
  return LoopOpKind::kNonLoopOp;
}

std::string LoopDirectionToStr(const LoopDirection& direction) {
  switch (direction) {
    case LoopDirection::kForward:
      return "forward";
    case LoopDirection::kReverse:
      return "reverse";
    case LoopDirection::kBidirectional:
      return "bidirectional";
    default:
      ORT_NOT_IMPLEMENTED("not implemented LoopDirection kind!");
  }
}

std::tuple<LoopDirection, int> MakeDirection(const Node& node, const LoopOpKind& loop_op_kind) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  std::string direction = "forward";

  if (loop_op_kind == LoopOpKind::kLoopOpLSTM) {
    ORT_ENFORCE(attrs.GetAttr("direction", &direction).IsOK());
  }

  if (direction == "forward") {
    return std::make_tuple(LoopDirection::kForward, 1);
  } else if (direction == "reverse") {
    return std::make_tuple(LoopDirection::kReverse, 1);
  } else if (direction == "bidirectional") {
    return std::make_tuple(LoopDirection::kBidirectional, 2);
  } else {
    ORT_THROW("Invalid 'direction' argument of '", direction,
              "'. Must be one of 'forward', 'reverse', or 'bidirectional'.");
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
