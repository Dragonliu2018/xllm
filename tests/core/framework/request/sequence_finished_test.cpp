/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "core/common/types.h"
#include "framework/block/block_manager_impl.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

Sequence make_decodable_sequence(const std::vector<int32_t>& prompt_token_ids,
                                 size_t max_generated_tokens,
                                 size_t seq_capacity,
                                 RequestSamplingParam* sampling_param,
                                 StoppingChecker* stopping_checker) {
  stopping_checker->set_max_generated_tokens(max_generated_tokens);

  SequenceParams params;
  params.seq_capacity = seq_capacity;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "sequence_finished_test";
  params.sampling_param = sampling_param;
  params.stopping_checker = stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_token_ids.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);
  return Sequence(/*index=*/0,
                  prompt_token_ids,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

void prepare_for_decode(Sequence* sequence, BlockManagerImpl* manager) {
  sequence->add_blocks(BlockType::KV, manager->allocate(4));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
}

}  // namespace

TEST(SequenceFinishedTest, FinishedStateRecomputesAndCaches) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  const std::vector<int32_t> prompt_token_ids = {10, 11, 12};
  Sequence sequence = make_decodable_sequence(prompt_token_ids,
                                              /*max_generated_tokens=*/3,
                                              /*seq_capacity=*/32,
                                              &sampling_param,
                                              &stopping_checker);
  prepare_for_decode(&sequence, &manager);

  EXPECT_FALSE(sequence.finished());

  sequence.append_token(Token(101));
  EXPECT_FALSE(sequence.finished());

  sequence.append_token(Token(102));
  EXPECT_FALSE(sequence.finished());

  sequence.append_token(Token(103));
  EXPECT_TRUE(sequence.finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::LENGTH);

  for (int32_t i = 0; i < 32; ++i) {
    EXPECT_TRUE(sequence.finished());
    EXPECT_EQ(sequence.finish_reason(), FinishReason::LENGTH);
  }
}

TEST(SequenceFinishedTest, ConcurrentFinishedQueryWithAppendToken) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(4);
  BlockManagerImpl manager(options);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  const std::vector<int32_t> prompt_token_ids = {10, 11, 12};
  Sequence sequence = make_decodable_sequence(prompt_token_ids,
                                              /*max_generated_tokens=*/512,
                                              /*seq_capacity=*/64,
                                              &sampling_param,
                                              &stopping_checker);
  prepare_for_decode(&sequence, &manager);

  const size_t initial_num_tokens = sequence.num_tokens();
  std::atomic<bool> writer_done{false};

  std::thread reader([&sequence, &writer_done]() {
    while (!writer_done.load(std::memory_order_acquire)) {
      (void)sequence.finished();
      (void)sequence.finish_reason();
    }
    for (int32_t i = 0; i < 1024; ++i) {
      (void)sequence.finished();
      (void)sequence.finish_reason();
    }
  });

  std::thread writer([&sequence, &writer_done]() {
    for (int32_t i = 0; i < 48; ++i) {
      sequence.append_token(Token(200 + i));
    }
    writer_done.store(true, std::memory_order_release);
  });

  writer.join();
  reader.join();

  // Verify final state: tokens were appended
  EXPECT_EQ(sequence.num_tokens(), initial_num_tokens + 48);

  // Verify consistency: finished() and finish_reason() agree
  const bool is_finished = sequence.finished();
  const FinishReason reason = sequence.finish_reason();
  if (is_finished) {
    EXPECT_NE(reason, FinishReason::NONE);
  } else {
    EXPECT_EQ(reason, FinishReason::NONE);
  }
}

}  // namespace xllm
