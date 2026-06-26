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

// MiMo-V2.5-ASR prompt processor.
//
// The prompt is expected to be pre-formatted by the Python serving layer:
//   - Text segments tokenized normally
//   - Audio positions marked with <|empty|> tokens (one per speech group)
//   - <|sosp|> / <|eosp|> wrapping the audio span
//
// This processor is a pass-through for the prompt string.  The actual
// <|empty|> → audio embedding replacement happens in
// MiMoAudioForCausalLMImpl::get_input_embeddings().

#pragma once

#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/prompt_processor.h"

namespace xllm {

class MiMoAudioPromptProcessor final : public PromptProcessor {
 public:
  explicit MiMoAudioPromptProcessor(const ModelArgs& /*args*/) {}

  void process(std::string& /*prompt*/, const MMData& /*mm_data*/) override {
    // Prompt is pre-formatted by the Python layer — no modification needed.
  }

  void find_mm_spans(const std::vector<int32_t>& /*token_ids*/,
                     MMData& /*mm_data*/) override {
    // Audio spans are identified by <|empty|> token positions at inference
    // time (see get_input_embeddings), not by span markers.
  }
};

}  // namespace xllm
