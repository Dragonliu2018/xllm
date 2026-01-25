/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include "core/layers/qwen2_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

class QWen2ModelImpl : public LlmModelImplBase<layer::Qwen2DecoderLayer> {
 public:
  QWen2ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen2DecoderLayer>("qwen2",
                                                   context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          model_args.hidden_size() / model_args.n_heads(),
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
      // Debug: log cos_sin_ cache info
      LOG(INFO) << "[QWen2Model] cos_sin_ cache created - shape: "
                << cos_sin_.sizes() << ", dtype: " << cos_sin_.dtype()
                << ", sample (pos 0 first 10): " << cos_sin_[0].slice(0, 0, 10)
                << ", sample (pos 1 first 10): " << cos_sin_[1].slice(0, 0, 10);
    }

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto layer = layer::Qwen2DecoderLayer(context);
      layers_.push_back(layer);
    }
  }
  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    // Use F::embedding instead of index() to correctly handle 2D positions
    // For positions [3, seq_len], F::embedding returns [3, seq_len, head_dim*2]
    namespace F = torch::nn::functional;

    auto target_cos_sin = F::embedding(positions, cos_sin_);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    // Debug: log shapes before reshape
    LOG(INFO) << "[apply_mrope] Before reshape - cos_pos shape: "
              << cos_pos.sizes()
              << ", positions.size(0): " << positions.size(0);

    auto apply = [this](torch::Tensor x) {
      auto sections = mrope_section_;
      sections.insert(sections.end(), sections.begin(), sections.end());

      auto vec = x.split(sections, -1);
      std::vector<torch::Tensor> selects;
      selects.reserve(vec.size());

      for (int64_t i = 0; i < vec.size(); ++i) {
        auto m = vec[i];
        selects.push_back(m[i % mrope_section_.size()]);
      }
      return torch::cat(selects, -1);
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));

    // Debug: log cos/sin after reorder for token 0 and token 36
    if (cos_pos.size(0) > 0) {
      auto cos0 = cos_pos[0].slice(0, 0, std::min(10L, cos_pos.size(-1)));
      auto sin0 = sin_pos[0].slice(0, 0, std::min(10L, sin_pos.size(-1)));
      LOG(INFO) << "[apply_mrope] after reorder shape: " << cos_pos.sizes()
                << ", token 0 cos[0:10]: " << cos0 << ", sin[0:10]: " << sin0;
      if (cos_pos.size(0) > 36) {
        auto cos36 = cos_pos[36].slice(0, 0, std::min(10L, cos_pos.size(-1)));
        auto sin36 = sin_pos[36].slice(0, 0, std::min(10L, sin_pos.size(-1)));
        LOG(INFO) << "[apply_mrope] token 36 positions: "
                  << positions.select(1, 36) << ", cos[0:10]: " << cos36
                  << ", sin[0:10]: " << sin36;
      }
    }

    return std::make_pair(cos_pos, sin_pos);
  }
};
TORCH_MODULE(QWen2Model);

class QWen2ForCausalLMImpl : public LlmForCausalLMImplBase<QWen2Model> {
 public:
  QWen2ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen2Model>(context) {}

  // Load state dict for model and lm_head components
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.get_dict_with_prefix("model."));
    if (tie_word_embeddings) {
      lm_head_->load_state_dict(
          state_dict.get_dict_with_prefix("model.embed_tokens."));
    } else {
      lm_head_->load_state_dict(state_dict.get_dict_with_prefix("lm_head."));
    }
  }
};
TORCH_MODULE(QWen2ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen2, QWen2ForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(qwen2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For Qwen2/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
