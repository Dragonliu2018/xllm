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

// MiMo-V2.5-ASR audio processor.
//
// Responsibility: wrap pre-computed RVQ codes into the MMDataItem format
// expected by MiMoAudioForCausalLM::get_multimodal_embeddings().
//
// ──────────────────────────────────────────────────────────────────────────────
// Why pre-computed?
//
// The RVQ audio tokenizer (MiMoAudioTokenizer) is a PyTorch neural network.
// Running it inside a C++ processor requires embedding a Python interpreter or
// re-implementing the model in C++—both are out of scope for this integration.
// Instead, the serving layer computes the codes in Python before submitting the
// request, and this processor simply validates and packages them.
//
// Convention: origin_audio is expected to have an INTEGER dtype
// (torch::kInt32 or torch::kInt64) and shape [T, audio_channels], where
// T = T_groups * group_size.  When it is a float tensor, it is treated as a
// raw waveform and an error is raised—waveform → code conversion must happen
// upstream.
// ──────────────────────────────────────────────────────────────────────────────

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/framework/model/model_args.h"
#include "processors/audio_processor.h"

namespace xllm {

class MiMoAudioProcessor final : public AudioProcessor {
 public:
  explicit MiMoAudioProcessor(const ModelArgs& args)
      : audio_channels_(args.audio_channels()),
        group_size_(args.group_size()) {}

  // origin_audio: pre-computed RVQ codes [T, audio_channels], integer dtype
  // metadata:     carries sample_rate / duration (informational, not used here)
  // output_item:  populated with key "audio|codes"
  bool process(const torch::Tensor& origin_audio,
               const AudioMetadata& /*metadata*/,
               MMDataItem& output_item) const override {
    CHECK(origin_audio.dtype() == torch::kInt32 ||
          origin_audio.dtype() == torch::kInt64)
        << "MiMoAudioProcessor expects integer RVQ codes, got dtype: "
        << origin_audio.dtype();
    CHECK_EQ(origin_audio.dim(), 2)
        << "Expected shape [T, audio_channels], got " << origin_audio.dim()
        << "D tensor";
    CHECK_EQ(origin_audio.size(1), audio_channels_)
        << "Expected " << audio_channels_ << " audio channels, got "
        << origin_audio.size(1);
    CHECK_EQ(origin_audio.size(0) % group_size_, 0)
        << "T=" << origin_audio.size(0)
        << " is not divisible by group_size=" << group_size_;

    output_item = MMDataItem(MMType::AUDIO);
    output_item.add<torch::Tensor>("audio|codes",
                                   origin_audio.to(torch::kInt32));
    return true;
  }

 private:
  int32_t audio_channels_;
  int32_t group_size_;
};

}  // namespace xllm
