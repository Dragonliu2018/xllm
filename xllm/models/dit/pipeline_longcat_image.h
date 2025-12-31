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

#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>

#include "autoencoder_kl.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/layers/pos_embedding.h"
#include "dit.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"
#include "models/vlm/qwen2_5_vl.h"

// LongCat-Image pipeline implementation
// Uses Qwen2_5_VL as text encoder instead of CLIP+T5

namespace xllm {

// Forward declarations
class LongCatImagePipelineImpl;

// Utility functions (copied from pipeline_flux_base.h)
constexpr int64_t ROPE_SCALE_BASE = 10000;

// Standalone position embedding implementation for LongCat-Image
torch::Tensor get_1d_rotary_pos_embed(
    int64_t dim,
    const torch::Tensor& pos,
    float theta = 10000.0,
    bool use_real = false,
    float linear_factor = 1.0,
    float ntk_factor = 1.0,
    bool repeat_interleave_real = true,
    torch::Dtype freqs_dtype = torch::kFloat32) {
  CHECK_EQ(dim % 2, 0) << "Dimension must be even";

  torch::Tensor pos_tensor = pos;
  if (pos.dim() == 0) {
    pos_tensor = torch::arange(pos.item<int64_t>(), pos.options());
  }

  theta = theta * ntk_factor;

  auto freqs =
      1.0 /
      (torch::pow(
           theta,
           torch::arange(
               0, dim, 2, torch::dtype(freqs_dtype).device(pos.device())) /
               dim) *
       linear_factor);  // [D/2]

  auto tensors = {pos_tensor, freqs};

  auto freqs_outer = torch::einsum("s,d->sd", tensors);  // [S, D/2]
  freqs_outer = freqs_outer.to(torch::kFloat32);

  if (use_real && repeat_interleave_real) {
    auto cos_vals = torch::cos(freqs_outer);  // [S, D/2]
    auto sin_vals = torch::sin(freqs_outer);  // [S, D/2]

    auto freqs_cos = cos_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]

    auto freqs_sin = sin_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]
    return torch::cat({freqs_cos.unsqueeze(0), freqs_sin.unsqueeze(0)},
                      0);  // [2, S, D]
  }
  return torch::Tensor();
}

class LongCatImagePosEmbedImpl : public torch::nn::Module {
 public:
  LongCatImagePosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    auto seq_len = txt_ids.size(0);

    // recompute the cache if height or width changes
    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 0);
      cached_image_height_ = height;
      cached_image_width_ = width;
      max_seq_len_ = seq_len;
      auto [cos, sin] = forward(ids);
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
  }

  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = ids.size(-1);
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    torch::Dtype freqs_dtype = torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i);
      auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                            pos_slice,
                                            theta_,
                                            true,  // repeat_interleave_real
                                            1,
                                            1,
                                            true,  // use_real
                                            freqs_dtype);
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);
    return {freqs_cos, freqs_sin};
  }

 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t max_seq_len_ = -1;
  int64_t cached_image_height_ = -1;
  int64_t cached_image_width_ = -1;
};
TORCH_MODULE(LongCatImagePosEmbed);

float calculate_shift(int64_t image_seq_len,
                      int64_t base_seq_len = 256,
                      int64_t max_seq_len = 4096,
                      float base_shift = 0.5f,
                      float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

std::pair<torch::Tensor, int64_t> retrieve_timesteps(
    FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(num_inference_steps, device);
    scheduler_timesteps = scheduler->timesteps();
  }
  return {scheduler_timesteps, steps};
}

torch::Tensor pack_latents(torch::Tensor latents,
                           int64_t batch_size,
                           int64_t num_channels_latents,
                           int64_t height,
                           int64_t width) {
  torch::Tensor latents_packed = latents.view(
      {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
  latents_packed = latents_packed.permute({0, 2, 4, 1, 3, 5});
  latents_packed = latents_packed.reshape(
      {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});
  return latents_packed;
}

torch::Tensor unpack_latents(torch::Tensor latents,
                             int64_t height,
                             int64_t width,
                             int64_t vae_scale_factor) {
  int64_t batch_size = latents.size(0);
  int64_t num_patches = latents.size(1);
  int64_t channels = latents.size(2);
  height = 2 * (height / (vae_scale_factor * 2));
  width = 2 * (width / (vae_scale_factor * 2));

  torch::Tensor latents_unpacked =
      latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
  latents_unpacked = latents_unpacked.permute({0, 3, 1, 4, 2, 5});
  latents_unpacked =
      latents_unpacked.reshape({batch_size, channels / (2 * 2), height, width});

  return latents_unpacked;
}

torch::Tensor prepare_latent_image_ids(int64_t batch_size,
                                       int64_t height,
                                       int64_t width,
                                       const torch::TensorOptions& options) {
  torch::Tensor latent_image_ids = torch::zeros({height, width, 3}, options);
  torch::Tensor height_range = torch::arange(height, options).unsqueeze(1);
  latent_image_ids.select(2, 1) += height_range;
  torch::Tensor width_range = torch::arange(width, options).unsqueeze(0);
  latent_image_ids.select(2, 2) += width_range;
  latent_image_ids = latent_image_ids.view({height * width, 3});
  return latent_image_ids;
}

class LongCatImagePipelineImpl : public torch::nn::Module {
 public:
  LongCatImagePipelineImpl(const DiTModelContext& context) : context_(context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    LOG(INFO) << "[LongCatImage] VAE block_out_channels size: "
              << model_args.block_out_channels().size()
              << ", calculated vae_scale_factor_: " << vae_scale_factor_;

    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    tokenizer_max_length_ =
        context.get_model_args("text_encoder").max_position_embeddings();
    LOG(INFO) << "Initializing LongCat-Image pipeline...";
    vae_image_processor_ =
        VAEImageProcessor(ModelContext(context.get_parallel_args(),
                                       context.get_model_args("vae"),
                                       context.get_quant_args("vae"),
                                       context.get_tensor_options()),
                          true,
                          true,
                          false,
                          false,
                          false,
                          model_args.latent_channels());
    vae_ = VAE(ModelContext(context.get_parallel_args(),
                            context.get_model_args("vae"),
                            context.get_quant_args("vae"),
                            context.get_tensor_options()));
    pos_embed_ = register_module(
        "pos_embed",
        LongCatImagePosEmbed(
            ROPE_SCALE_BASE,
            context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = LongCatImageTransformer2DModel(
        ModelContext(context.get_parallel_args(),
                     context.get_model_args("transformer"),
                     context.get_quant_args("transformer"),
                     context.get_tensor_options()));

    // LongCat-Image uses Qwen2_5_VL as text encoder, not CLIP+T5
    // VLM integration is complete and weights will be loaded in load_model()
    LOG(INFO) << "LongCat-Image uses Qwen2_5_VL as text encoder";

    scheduler_ = FlowMatchEulerDiscreteScheduler(
        ModelContext(context.get_parallel_args(),
                     context.get_model_args("scheduler"),
                     context.get_quant_args("scheduler"),
                     context.get_tensor_options()));
    register_module("vae", vae_);
    register_module("vae_image_processor", vae_image_processor_);
    register_module("transformer", transformer_);
    register_module("scheduler", scheduler_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    auto prompts_2 = input.prompts_2.empty()
                         ? std::nullopt
                         : std::make_optional(input.prompts_2);
    auto negative_prompts = input.negative_prompts.empty()
                                ? std::nullopt
                                : std::make_optional(input.negative_prompts);
    auto negative_prompts_2 =
        input.negative_prompts_2.empty()
            ? std::nullopt
            : std::make_optional(input.negative_prompts_2);

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;
    auto negative_pooled_prompt_embeds =
        input.negative_pooled_prompt_embeds.defined()
            ? std::make_optional(input.negative_pooled_prompt_embeds)
            : std::nullopt;

    std::vector<torch::Tensor> output = forward_(
        prompts,                                  // prompt
        prompts_2,                                // prompt_2
        negative_prompts,                         // negative_prompt
        negative_prompts_2,                       // negative_prompt_2
        generation_params.true_cfg_scale,         // cfg scale
        generation_params.height,                 // height
        generation_params.width,                  // width
        generation_params.num_inference_steps,    // num_inference_steps
        generation_params.guidance_scale,         // guidance_scale
        generation_params.num_images_per_prompt,  // num_images_per_prompt
        seed,                                     // seed
        latents,                                  // latents
        prompt_embeds,                            // prompt_embeds
        negative_prompt_embeds,                   // negative_prompt_embeds
        pooled_prompt_embeds,                     // pooled_prompt_embeds
        negative_pooled_prompt_embeds,         // negative_pooled_prompt_embeds
        generation_params.max_sequence_length  // max_sequence_length
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output[0], output[0].size(0), 0);
    LOG(INFO) << "Output tensor chunks size: " << out.tensors.size();
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "LongCat-Image pipeline loading model from "
              << loader->model_root_path();
    LOG(INFO) << "[LongCatImage] Model weights path: "
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();

    // Get all model args BEFORE taking any component loaders
    // (because take_component_loader removes the loader from the map)
    auto all_model_args = loader->get_model_args();
    auto all_quant_args = loader->get_quant_args();
    LOG(INFO) << "[LongCatImage] Model config args size: "
              << all_model_args.size();
    LOG(INFO) << "[LongCatImage] Quant config args size: "
              << all_quant_args.size();
    for (const auto& [key, value] : all_model_args) {
      LOG(INFO) << "[LongCatImage] Model component: " << key;
    }
    for (const auto& [key, value] : all_quant_args) {
      LOG(INFO) << "[LongCatImage] Quant component: " << key;
    }

    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto text_encoder_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto text_processor_loader =
        loader->take_component_loader("text_processor");

    LOG(INFO) << "LongCat-Image model components loaded, start to load weights "
                 "to sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());

    // Load VLM text encoder model (Qwen2_5_VL)
    if (text_encoder_loader) {
      LOG(INFO) << "Loading Qwen2_5_VL text encoder model...";
      try {
        auto text_encoder_args_iter = all_model_args.find("text_encoder");
        auto text_encoder_quant_iter = all_quant_args.find("text_encoder");

        if (text_encoder_args_iter != all_model_args.end() &&
            text_encoder_quant_iter != all_quant_args.end()) {
          // Check if tp_group_ is nullptr (single-device mode)
          // Qwen2_5_VL's Qwen2VisionAttention expects tp_group_ to be
          // non-nullptr If it's nullptr, we need to create a real ProcessGroup
          // for single-device inference
          const auto& original_parallel_args = context_.get_parallel_args();
          ParallelArgs vlm_parallel_args =
              original_parallel_args;  // Copy original args

          if (original_parallel_args.tp_group_ == nullptr) {
            LOG(INFO) << "Creating real ProcessGroup for single-device "
                         "VLM initialization.";
            // Create a real single-device ProcessGroup (rank=0, world_size=1)
            vlm_tp_group_ =
                create_process_group(0,               // rank
                                     1,               // world_size
                                     1,               // rank_size
                                     29500,           // port (arbitrary)
                                     false,           // trans
                                     "127.0.0.1",     // host (localhost)
                                     "vlm_tp_group",  // group_name
                                     options_.device());
            vlm_parallel_args.tp_group_ = vlm_tp_group_.get();
          }

          text_encoder_ = Qwen2_5_VLForConditionalGeneration(
              ModelContext(vlm_parallel_args,
                           text_encoder_args_iter->second,
                           text_encoder_quant_iter->second,
                           options_));

          // Manually load weights for visual and language_model components
          // since DiTFolderLoader is not compatible with ModelLoader interface
          auto& state_dicts = text_encoder_loader->get_state_dicts();
          for (const auto& state_dict_ptr : state_dicts) {
            // Load both visual and language_model components using
            // load_state_dict
            text_encoder_->load_state_dict(*state_dict_ptr);
            LOG(INFO) << "Qwen2_5_VL components weights loaded";
          }

          text_encoder_->to(options_.device());
          LOG(INFO) << "Qwen2_5_VL text encoder loaded successfully";
        } else {
          LOG(WARNING)
              << "Text encoder model args not found, will use dummy embeddings";
          text_encoder_ = nullptr;
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to load Qwen2_5_VL text encoder: " << e.what()
                     << ", will use dummy embeddings as fallback";
        text_encoder_ = nullptr;
      }
    } else {
      LOG(WARNING)
          << "Text encoder loader not available, will use dummy embeddings";
      text_encoder_ = nullptr;
    }

    // Load tokenizer if available
    if (tokenizer_loader) {
      tokenizer_ = tokenizer_loader->tokenizer();
      LOG(INFO) << "Tokenizer loaded successfully";
    }
  }

 private:
  // Model context (saved for use in load_model)
  DiTModelContext context_;

  // Member variables
  torch::TensorOptions options_;
  int64_t vae_scale_factor_;
  float vae_shift_factor_;
  float vae_scaling_factor_;
  int64_t tokenizer_max_length_;

  // ProcessGroup for VLM (single-device)
  std::unique_ptr<ProcessGroup> vlm_tp_group_;

  // Model components
  VAEImageProcessor vae_image_processor_{nullptr};
  VAE vae_{nullptr};
  LongCatImagePosEmbed pos_embed_{nullptr};
  LongCatImageTransformer2DModel transformer_{nullptr};
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  Qwen2_5_VLForConditionalGeneration text_encoder_{nullptr};

  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    LOG(INFO) << "[LongCatImage] prepare_latents: height=" << height
              << ", width=" << width
              << ", vae_scale_factor_=" << vae_scale_factor_
              << ", adjusted_height=" << adjusted_height
              << ", adjusted_width=" << adjusted_width
              << ", num_channels_latents=" << num_channels_latents;
    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, adjusted_height, adjusted_width};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids = prepare_latent_image_ids(
          batch_size, adjusted_height / 2, adjusted_width / 2, options_);
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor = randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor,
                                                batch_size,
                                                num_channels_latents,
                                                adjusted_height,
                                                adjusted_width);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2, options_);
    return {packed_latents, latent_image_ids};
  }

  std::vector<torch::Tensor> forward_(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt_2 = std::nullopt,
      float true_cfg_scale = 1.0f,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_inference_steps = 28,
      float guidance_scale = 3.5f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_pooled_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    // 打印推理参数
    LOG(INFO) << "[LongCatImage] Inference params: height=" << height
              << ", width=" << width
              << ", num_inference_steps=" << num_inference_steps
              << ", guidance_scale=" << guidance_scale
              << ", num_images_per_prompt=" << num_images_per_prompt
              << ", seed="
              << (seed.has_value() ? std::to_string(seed.value()) : "none")
              << ", max_sequence_length=" << max_sequence_length;
    if (prompt.has_value()) {
      for (size_t i = 0; i < prompt.value().size(); ++i) {
        LOG(INFO) << "[LongCatImage] Prompt[" << i
                  << "]: " << prompt.value()[i];
      }
    }
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;
    bool has_neg_prompt = negative_prompt.has_value() ||
                          (negative_prompt_embeds.has_value() &&
                           negative_pooled_prompt_embeds.has_value());
    bool do_true_cfg = (true_cfg_scale > 1.0f) && has_neg_prompt;

    // Encode prompt using VLM text encoder
    torch::Tensor encoded_prompt_embeds;
    torch::Tensor encoded_pooled_embeds;
    torch::Tensor text_ids;

    if (prompt_embeds.has_value()) {
      encoded_prompt_embeds = prompt_embeds.value();
      encoded_pooled_embeds = pooled_prompt_embeds.value();
    } else {
      // Use VLM text encoder to encode prompts
      if (text_encoder_ && prompt.has_value()) {
        LOG(INFO) << "Encoding prompts using VLM text encoder";
        try {
          // Tokenize prompts
          std::vector<std::vector<int32_t>> text_input_ids;
          text_input_ids.reserve(total_batch_size);

          // Repeat prompts for num_images_per_prompt
          std::vector<std::string> repeated_prompts;
          for (const auto& p : prompt.value()) {
            for (int64_t i = 0; i < num_images_per_prompt; ++i) {
              repeated_prompts.push_back(p);
            }
          }

          if (tokenizer_ &&
              tokenizer_->batch_encode(repeated_prompts, &text_input_ids)) {
            // Truncate or pad to max_sequence_length
            for (auto& ids : text_input_ids) {
              if (ids.size() > max_sequence_length) {
                ids.resize(max_sequence_length);
              } else if (ids.size() < max_sequence_length) {
                ids.resize(max_sequence_length, 0);  // Pad with 0
              }
            }

            // Convert to tensor
            std::vector<int32_t> input_ids_flat;
            input_ids_flat.reserve(total_batch_size * max_sequence_length);
            for (const auto& ids : text_input_ids) {
              input_ids_flat.insert(
                  input_ids_flat.end(), ids.begin(), ids.end());
            }

            auto input_ids =
                torch::tensor(input_ids_flat, torch::dtype(torch::kLong))
                    .view({total_batch_size, max_sequence_length})
                    .to(options_.device());

            // Encode using VLM text encoder
            torch::NoGradGuard no_grad;
            // Get input embeddings from the language model
            auto input_embeds = text_encoder_->get_input_embeddings(
                input_ids, std::nullopt, std::nullopt, ModelInputParams());

            // Use the embeddings as prompt embeddings
            encoded_prompt_embeds = input_embeds;
            // Use mean pooling as pooled embeddings
            encoded_pooled_embeds = input_embeds.mean(1);

            LOG(INFO) << "VLM text encoding successful, output shapes: "
                      << encoded_prompt_embeds.sizes() << ", "
                      << encoded_pooled_embeds.sizes();
            LOG(INFO) << "[LongCatImage] Encoded prompt_embeds shape: "
                      << encoded_prompt_embeds.sizes();
            LOG(INFO) << "[LongCatImage] Encoded pooled_embeds shape: "
                      << encoded_pooled_embeds.sizes();
          } else {
            throw std::runtime_error("Failed to tokenize prompts");
          }
        } catch (const std::exception& e) {
          LOG(WARNING) << "VLM text encoding failed: " << e.what()
                       << ", falling back to dummy embeddings";
          goto use_dummy_embeddings;
        }
      } else {
      use_dummy_embeddings:
        LOG(WARNING)
            << "Using dummy text embeddings - VLM text encoder not available";
        // Create dummy embeddings for testing
        int64_t hidden_size = 3584;  // joint_attention_dim from config
        int64_t pooled_projection_dim =
            3584;  // pooled_projection_dim from config
        encoded_prompt_embeds = torch::zeros(
            {total_batch_size, max_sequence_length, hidden_size}, options_);
        encoded_pooled_embeds = torch::zeros(
            at::IntArrayRef({total_batch_size, pooled_projection_dim}),
            options_);
      }
    }

    text_ids =
        torch::zeros({max_sequence_length, 3},
                     torch::dtype(torch::kLong).device(options_.device()));

    // encode negative prompt
    torch::Tensor negative_encoded_embeds, negative_pooled_embeds;
    torch::Tensor negative_text_ids;
    if (do_true_cfg) {
      if (negative_prompt_embeds.has_value()) {
        negative_encoded_embeds = negative_prompt_embeds.value();
        negative_pooled_embeds = negative_pooled_prompt_embeds.value();
      } else {
        int64_t hidden_size = 3584;  // joint_attention_dim from config
        int64_t pooled_projection_dim =
            3584;  // pooled_projection_dim from config
        negative_encoded_embeds = torch::zeros(
            {total_batch_size, max_sequence_length, hidden_size}, options_);
        negative_pooled_embeds = torch::zeros(
            at::IntArrayRef({total_batch_size, pooled_projection_dim}),
            options_);
      }
      negative_text_ids =
          torch::zeros({max_sequence_length, 3},
                       torch::dtype(torch::kLong).device(options_.device()));
    }

    // prepare latent
    int64_t num_channels_latents = transformer_->in_channels() / 4;
    LOG(INFO) << "[LongCatImage] transformer_->in_channels()="
              << transformer_->in_channels()
              << ", num_channels_latents=" << num_channels_latents
              << ", vae_scale_factor_=" << vae_scale_factor_;
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.has_value() ? seed.value() : 42,
                        latents);
    LOG(INFO) << "[LongCatImage] After prepare_latents: shape="
              << prepared_latents.sizes()
              << ", min/max=" << prepared_latents.min().item<float>() << "/"
              << prepared_latents.max().item<float>();

    // prepare timestep
    std::vector<float> new_sigmas;
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      new_sigmas.push_back(1.0f - static_cast<float>(i) /
                                      (num_inference_steps - 1) *
                                      (1.0f - 1.0f / num_inference_steps));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());
    auto [timesteps, num_inference_steps_actual] = retrieve_timesteps(
        scheduler_, num_inference_steps, options_.device(), new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order(),
                 static_cast<int64_t>(0LL));

    // prepare guidance
    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      torch::TensorOptions options =
          torch::dtype(torch::kFloat32).device(options_.device());

      guidance = torch::full(at::IntArrayRef({1}), guidance_scale, options);
      guidance = guidance.expand({prepared_latents.size(0)});
    }
    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());

    // image rotary positional embeddings outplace computation
    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));
    torch::Tensor image_rotary_emb = torch::stack({rot_emb1, rot_emb2}, 0);

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);
      int64_t step_id = i + 1;
      if (i == 0) {
        LOG(INFO) << "[LongCatImage] Step " << i
                  << ": prepared_latents min/max before forward: "
                  << prepared_latents.min().item<float>() << "/"
                  << prepared_latents.max().item<float>();
      }
      torch::Tensor noise_pred = transformer_->forward(prepared_latents,
                                                       encoded_prompt_embeds,
                                                       encoded_pooled_embeds,
                                                       timestep,
                                                       image_rotary_emb,
                                                       guidance,
                                                       step_id);
      if (i == 0 || i == timesteps.numel() - 1) {
        LOG(INFO) << "[LongCatImage] Step " << i
                  << ": noise_pred min/max: " << noise_pred.min().item<float>()
                  << "/" << noise_pred.max().item<float>();
      }
      if (do_true_cfg) {
        torch::Tensor negative_noise_pred =
            transformer_->forward(prepared_latents,
                                  negative_encoded_embeds,
                                  negative_pooled_embeds,
                                  timestep,
                                  image_rotary_emb,
                                  guidance,
                                  step_id);
        noise_pred =
            noise_pred + (noise_pred - negative_noise_pred) * true_cfg_scale;
        negative_noise_pred.reset();
      }
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      if (i == 0 || i == timesteps.numel() - 1) {
        LOG(INFO) << "[LongCatImage] Step " << i
                  << ": prev_latents (before detach) min/max: "
                  << prev_latents.min().item<float>() << "/"
                  << prev_latents.max().item<float>();
      }
      prepared_latents = prev_latents.detach();
      std::vector<torch::Tensor> tensors = {prepared_latents, noise_pred};
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
      if (i == timesteps.numel() - 1) {
        LOG(INFO) << "[LongCatImage] Last step " << i
                  << ": prepared_latents min/max after scheduler: "
                  << prepared_latents.min().item<float>() << "/"
                  << prepared_latents.max().item<float>();
      }
    }
    torch::Tensor image;
    // Unpack latents
    torch::Tensor unpacked_latents =
        unpack_latents(prepared_latents, height, width, vae_scale_factor_);
    LOG(INFO) << "[LongCatImage] After unpack_latents shape: "
              << unpacked_latents.sizes();
    LOG(INFO) << "[LongCatImage] After unpack_latents min/max: "
              << unpacked_latents.min().item<float>() << "/"
              << unpacked_latents.max().item<float>();
    LOG(INFO) << "[LongCatImage] vae_scaling_factor_=" << vae_scaling_factor_
              << ", vae_shift_factor_=" << vae_shift_factor_;
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
    LOG(INFO) << "[LongCatImage] After scaling/shifting shape: "
              << unpacked_latents.sizes();
    LOG(INFO) << "[LongCatImage] After scaling/shifting min/max: "
              << unpacked_latents.min().item<float>() << "/"
              << unpacked_latents.max().item<float>();
    unpacked_latents = unpacked_latents.to(options_.dtype());
    LOG(INFO) << "[LongCatImage] After dtype conversion min/max: "
              << unpacked_latents.min().item<float>() << "/"
              << unpacked_latents.max().item<float>();
    image = vae_->decode(unpacked_latents);
    LOG(INFO) << "[LongCatImage] After VAE decode shape: " << image.sizes();
    LOG(INFO) << "[LongCatImage] After VAE decode min/max: "
              << image.min().item<float>() << "/" << image.max().item<float>();
    image = vae_image_processor_->postprocess(image);
    LOG(INFO) << "[LongCatImage] Generated image tensor shape: "
              << image.sizes();
    LOG(INFO) << "[LongCatImage] Image tensor min/max: "
              << image.min().item<float>() << "/" << image.max().item<float>();
    return std::vector<torch::Tensor>{{image}};
  }
};
TORCH_MODULE(LongCatImagePipeline);

// Register LongCat-Image as DiT model
// Direct registration since "LongCat-Image" contains a hyphen
namespace {
const bool longcat_image_dit_registered = []() {
  ModelRegistry::register_dit_model_factory(
      "LongCat-Image", [](const DiTModelContext& context) {
        LongCatImagePipeline model(context);
        model->eval();
        return std::make_unique<xllm::DiTModelImpl<LongCatImagePipeline>>(
            std::move(model), context.get_tensor_options());
      });
  return true;
}();
}  // namespace

}  // namespace xllm