/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <unordered_map>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/lm_head.h"
#include "core/layers/qwen2_decoder_layer.h"
#include "core/layers/qwen2dot5_vision_encode_layer.h"
#include "models/llm/qwen2.h"
#include "models/model_registry.h"
#include "models/vlm/qwen2_5_vl.h"
#include "processors/input_processor.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {

#define PrintTensor(tensor) print_tensor(tensor, #tensor, 10, true, false);

// Reuse Qwen2_5_VLInputProcessor for LongCat-Image
using LongCatImageInputProcessor = Qwen2_5_VLInputProcessor;

struct LongCatImageImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct LongCatImageVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class LongCatImageForConditionalGenerationImpl : public torch::nn::Module {
 public:
  LongCatImageForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_5_VisionTransformer(context));

    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  torch::Tensor get_input_embeddings(
      torch::Tensor input_ids,
      const std::optional<LongCatImageImageInputs>& image_input,
      const std::optional<LongCatImageVideoInputs>& video_input,
      const ModelInputParams& input_params) {
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (image_input) {
      // visual
      auto image_embeds = visual_(image_input->pixel_values.to(options_),
                                  image_input->image_grid_thw,
                                  input_params);
      // merge
      auto is_multimodal = torch::isin(input_ids, model_args_.image_token_id());
      inputs_embeds.index_put_({is_multimodal}, image_embeds);
    }
    if (video_input) {
      // visual
      auto video_embeds = visual_(video_input->pixel_values_videos.to(options_),
                                  video_input->video_grid_thw,
                                  input_params);
      // merge
      auto is_multimodal = torch::isin(input_ids, model_args_.video_token_id());
      inputs_embeds.index_put_({is_multimodal}, video_embeds);
    }
    return inputs_embeds;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;

    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor second_per_grid_ts;
    if (const auto& res = mm_data.get<torch::Tensor>("second_per_grid_ts"))
      second_per_grid_ts = res.value();

    std::optional<LongCatImageImageInputs> image_inputs;
    std::optional<LongCatImageVideoInputs> video_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = LongCatImageImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined() &&
        second_per_grid_ts.defined())
      video_inputs = LongCatImageVideoInputs{
          pixel_values_videos, video_grid_thw, second_per_grid_ts};

    auto inputs_embeds =
        get_input_embeddings(tokens, image_inputs, video_inputs, input_params);
    input_params.input_embedding = inputs_embeds;

    auto emb = language_model_(tokens, positions, kv_caches, input_params);

    return emb;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader));
    }
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }

  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_5_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(LongCatImageForConditionalGeneration);

// Register with underscore naming
REGISTER_INPUT_PROCESSOR(longcat_image, LongCatImageInputProcessor);
REGISTER_CAUSAL_VLM_MODEL(longcat_image, LongCatImageForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(longcat_image, Qwen2VLImageProcessor);

// Also register with hyphen naming (LongCat-Image) for compatibility
// Note: We need to use direct registration since "LongCat-Image" contains a
// hyphen which is not a valid C++ identifier
namespace {
const bool longcat_image_hyphen_input_processor_registered = []() {
  ModelRegistry::register_input_processor_factory(
      "LongCat-Image", [](const ModelArgs& args) {
        return std::make_unique<LongCatImageInputProcessor>(args);
      });
  return true;
}();

const bool longcat_image_hyphen_vlm_registered = []() {
  ModelRegistry::register_causalvlm_factory(
      "LongCat-Image", [](const ModelContext& context) {
        LongCatImageForConditionalGeneration model(context);
        model->eval();
        return std::make_unique<
            xllm::CausalVLMImpl<LongCatImageForConditionalGeneration>>(
            std::move(model), context.get_tensor_options());
      });
  // register_causalvlm_factory already sets backend to "vlm", so no need to set
  // it again
  return true;
}();

const bool longcat_image_hyphen_image_processor_registered = []() {
  ModelRegistry::register_image_processor_factory(
      "LongCat-Image", [](const ModelArgs& args) {
        return std::make_unique<Qwen2VLImageProcessor>(args);
      });
  return true;
}();
}  // namespace

REGISTER_MODEL_ARGS(longcat_image, [&] {
  // text config
  // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 128000);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(model_type, "model_type", "longcat_image");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
  // LOAD_ARG_OR(use_cache, "use_cache", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1280);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 3420);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
  LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);
  LOAD_ARG_OR(mm_fullatt_block_indexes,
              "vision_config.fullatt_block_indexes",
              std::vector<int64_t>({7, 15, 23, 31}));
  LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(
      rope_scaling_rope_type, "vision_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
});

// Also register model args loader for LongCat-Image (with hyphen)
// Note: Direct registration since "LongCat-Image" contains a hyphen
namespace {
const bool longcat_image_hyphen_args_registered = []() {
  ModelRegistry::register_model_args_loader(
      "LongCat-Image", [](const JsonReader& json, ModelArgs* args) {
        UNUSED_PARAMETER(json);
        UNUSED_PARAMETER(args);
        // text config
        // LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0);
        LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
        LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
        LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
        LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
        LOAD_ARG_OR(vision_token_id, "vision_token_id", 151654);
        LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
        LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
        LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
        LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
        // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
        LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
        LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 128000);
        LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
        LOAD_ARG_OR(model_type, "model_type", "LongCat-Image");
        LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
        LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
        LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
        LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
        LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
        LOAD_ARG_OR(sliding_window, "sliding_window", 32768);
        LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
        LOAD_ARG_OR(dtype, "torch_dtype", "");
        // LOAD_ARG_OR(transformers_version, "transformers_version", "4.41.2");
        // LOAD_ARG_OR(use_cache, "use_cache", true);
        LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
        LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
          return args->hidden_size() / args->n_heads();
        });

        // vision_config
        LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 32);
        LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "silu");
        LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1280);
        LOAD_ARG_OR(
            mm_intermediate_size, "vision_config.intermediate_size", 3420);
        LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
        LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
        LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
        LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
        LOAD_ARG_OR(
            mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
        LOAD_ARG_OR(
            mm_spatial_patch_size, "vision_config.spatial_patch_size", 14);
        LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);
        LOAD_ARG_OR(mm_fullatt_block_indexes,
                    "vision_config.fullatt_block_indexes",
                    std::vector<int64_t>({7, 15, 23, 31}));
        LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);
        LOAD_ARG_OR(
            mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
        LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
          return args->mm_hidden_size() / args->mm_num_attention_heads();
        });

        LOAD_ARG_OR(
            rope_scaling_rope_type, "vision_config.rope_scaling.type", "mrope");
        LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");
        LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
        return true;
      });
  return true;
}();
}  // namespace

}  // namespace xllm
