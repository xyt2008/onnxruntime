// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string.h>
#include <tuple>
#include <vector>

namespace onnxruntime {

class Node;

namespace tvm_codegen {

// A enum class to avoid string-comparing on Node::OpType
enum class LoopOpKind {
  kNonLoopOp,

  // TODO: add others
  kLoopOpGRU,
  kLoopOpLSTM,
  kLoopOpRNN,
  kLoopOpScan,

  kUnknown,
};

LoopOpKind GetLoopOpKind(const Node& node);

// TODO: consider to refactor this along with rnn::detail::Direction in rnn_helpers.h
//       by moving the definition into a common header file.
enum class LoopDirection {
  kForward = 0,
  kReverse = 1,
  kBidirectional = 2,
};

std::string LoopDirectionToStr(const LoopDirection& direction);

std::tuple<LoopDirection, int> MakeDirection(const Node& node, const LoopOpKind&);

// TODO: we may not need to actually reverse data.
// Try to implement an iterator style reversing mechanism.
template <typename T>
void ReverseSequenceData(T* dest_data,
                         const T* src_data,
                         const std::vector<int>& sequence_lens,
                         int max_sequence_length,
                         int64_t batch_size,
                         int64_t input_size,
                         int num_directions) {
  int size_per_seq = batch_size * input_size;
  for (int i = 0; i < batch_size; ++i) {
    int seq_len = sequence_lens[i];
    // only reverse valid content of the sequence
    for (int j = 0; j < seq_len; j++) {
      int64_t src_offset = j * size_per_seq + i * input_size;
      int64_t dest_offset = num_directions * (seq_len - j - 1) * size_per_seq + i * input_size;
      memcpy(dest_data + dest_offset, src_data + src_offset, input_size * sizeof(T));
    }

    // add previously dummy content to the end
    for (int j = seq_len; j < max_sequence_length; j++) {
      int64_t src_offset = j * size_per_seq + i * input_size;
      int64_t dest_offset = num_directions * j * size_per_seq + i * input_size;
      memcpy(dest_data + dest_offset, src_data + src_offset, input_size * sizeof(T));
    }
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
