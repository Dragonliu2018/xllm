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
#include "core/layers/common/pos_embedding.h"
#include "core/layers/cuda/flashinfer_workspace.h"
#include "dit.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"
#include "models/vlm/qwen2_5_vl.h"

namespace xllm {

// Forward declarations
class LongCatImagePipelineImpl;

// Utility constants
constexpr int64_t ROPE_SCALE_BASE = 10000;

// Prompt template constants for LongCat-Image
// Ref:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/longcat_image/pipeline_longcat_image.py
constexpr const char* PROMPT_TEMPLATE_ENCODE_PREFIX =
    "<|im_start|>system\nAs an image captioning expert, generate a descriptive "
    "text prompt based on an image content, suitable for input to a "
    "text-to-image model.<|im_end|>\n<|im_start|>user\n";
constexpr const char* PROMPT_TEMPLATE_ENCODE_SUFFIX =
    "<|im_end|>\n<|im_start|>assistant\n";

// get_1d_rotary_pos_embed and LongCatImagePosEmbed are defined in dit.h

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

torch::Tensor prepare_latent_image_ids(int64_t height,
                                       int64_t width,
                                       int64_t start_h,
                                       int64_t start_w,
                                       const torch::TensorOptions& options) {
  torch::Tensor latent_image_ids = torch::zeros({height, width, 3}, options);
  // Set modality_id for image (1)
  latent_image_ids.select(2, 0).fill_(1);
  // Set position indices with start offset
  torch::Tensor height_range = torch::arange(height, options).unsqueeze(1);
  latent_image_ids.select(2, 1) += height_range + start_h;
  torch::Tensor width_range = torch::arange(width, options).unsqueeze(0);
  latent_image_ids.select(2, 2) += width_range + start_w;
  latent_image_ids = latent_image_ids.view({height * width, 3});

  // Log position IDs for verification
  LOG(INFO) << "[LongCatImage] Image position IDs (global) - start_h: "
            << start_h << ", start_w: " << start_w << ", first few IDs: "
            << latent_image_ids.slice(0, 0, std::min(3L, height * width));

  return latent_image_ids;
}

class LongCatImagePipelineImpl : public torch::nn::Module {
 public:
  LongCatImagePipelineImpl(const DiTModelContext& context) : context_(context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();

    // Initialize FlashinferWorkspace for attention operations
    // This is required for batch_prefill to work correctly
    xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
        options_.device());

    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    LOG(INFO) << "[LongCatImage] VAE block_out_channels size: "
              << model_args.block_out_channels().size()
              << ", calculated vae_scale_factor_: " << vae_scale_factor_;

    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    LOG(INFO) << "[LongCatImage] VAE parameters - scaling_factor: "
              << vae_scaling_factor_ << ", shift_factor: " << vae_shift_factor_
              << ", scale_factor: " << vae_scale_factor_;
    tokenizer_max_length_ =
        context.get_model_args("text_encoder").max_position_embeddings();
    LOG(INFO) << "[LongCatImage] tokenizer_max_length_ (from "
                 "max_position_embeddings): "
              << tokenizer_max_length_
              << ", but image position IDs will use hardcoded 512 (matching "
                 "Python diffusers)";
    LOG(INFO) << "Initializing LongCat-Image pipeline...";
    vae_image_processor_ = VAEImageProcessor(
        ModelContext(context.get_parallel_args(),
                     context.get_model_args("vae"),
                     context.get_quant_args("vae"),
                     context.get_tensor_options()),
        true,  // do_resize
        true,  // do_normalize (denormalize VAE output from [-1, 1] to [0, 1])
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

    // 判断是否需要创建单卡 ProcessGroup，并挂载到 ParallelArgs
    const auto& original_parallel_args = context.get_parallel_args();
    ParallelArgs vlm_parallel_args = original_parallel_args;  // 拷贝
    if (original_parallel_args.tp_group_ == nullptr) {
      LOG(INFO)
          << "Creating real ProcessGroup for single-device VLM initialization.";
      vlm_tp_group_ = create_process_group(0,
                                           1,
                                           1,
                                           29500,
                                           false,
                                           "127.0.0.1",
                                           "vlm_tp_group",
                                           options_.device());
      vlm_parallel_args.tp_group_ = vlm_tp_group_.get();
    }

    // LongCat-Image uses Qwen2_5_VL as text encoder, not CLIP+T5
    LOG(INFO) << "LongCat-Image uses Qwen2_5_VL as text encoder";
    // 初始化 text_encoder_（只构造对象，不加载权重），传入新的
    // vlm_parallel_args
    text_encoder_ = Qwen2_5_VLForConditionalGeneration(
        ModelContext(vlm_parallel_args,
                     context.get_model_args("text_encoder"),
                     context.get_quant_args("text_encoder"),
                     context.get_tensor_options()));

    scheduler_ = FlowMatchEulerDiscreteScheduler(
        ModelContext(context.get_parallel_args(),
                     context.get_model_args("scheduler"),
                     context.get_quant_args("scheduler"),
                     context.get_tensor_options()));
    // Prompt template for encoding
    prompt_template_encode_prefix_ = std::string(PROMPT_TEMPLATE_ENCODE_PREFIX);
    prompt_template_encode_suffix_ = std::string(PROMPT_TEMPLATE_ENCODE_SUFFIX);

    register_module("vae", vae_);
    register_module("vae_image_processor", vae_image_processor_);
    register_module("transformer", transformer_);
    register_module("text_encoder", text_encoder_);
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
        negative_pooled_prompt_embeds,          // negative_pooled_prompt_embeds
        generation_params.max_sequence_length,  // max_sequence_length
        generation_params.enable_cfg_renorm,    // enable_cfg_renorm
        generation_params.cfg_renorm_min        // cfg_renorm_min
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

    // Load and verify Transformer
    LOG(INFO) << "[WEIGHT_CHECK] Starting Transformer weight loading...";
    LOG(INFO) << "[WEIGHT_CHECK] Transformer config: "
              << "num_layers="
              << context_.get_model_args("transformer").num_layers()
              << ", num_single_layers="
              << context_.get_model_args("transformer").num_single_layers()
              << ", joint_attention_dim="
              << context_.get_model_args("transformer").joint_attention_dim()
              << ", pooled_projection_dim="
              << context_.get_model_args("transformer").pooled_projection_dim()
              << ", guidance_embeds="
              << context_.get_model_args("transformer").guidance_embeds();
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    transformer_->eval();  // Set transformer to evaluation mode
    // Verify loaded weights including NaN checks
    transformer_->verify_loaded_weights("transformer.");
    LOG(INFO) << "[WEIGHT_CHECK] Transformer weight loading completed.";

    // Load and verify VAE
    LOG(INFO) << "[WEIGHT_CHECK] Starting VAE weight loading...";
    LOG(INFO) << "[WEIGHT_CHECK] VAE config: "
              << "latent_channels="
              << context_.get_model_args("vae").latent_channels()
              << ", scaling_factor="
              << context_.get_model_args("vae").scale_factor()
              << ", shift_factor="
              << context_.get_model_args("vae").shift_factor();
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    vae_->eval();  // Set VAE to evaluation mode
    LOG(INFO) << "[WEIGHT_CHECK] VAE weight loading completed.";

    // Load VLM text encoder model (Qwen2_5_VL)
    if (text_encoder_loader) {
      LOG(INFO) << "Loading Qwen2_5_VL text encoder weights...";
      try {
        // 使用 load_model 方法加载权重
        // 这会自动处理 visual 和 language_model 的权重加载
        // DiTFolderLoader 继承自 ModelLoader，可以直接转换
        LOG(INFO) << "Before load_model: text_encoder_ is initialized";
        std::unique_ptr<ModelLoader> model_loader(
            std::move(text_encoder_loader));
        text_encoder_->load_model(std::move(model_loader));
        LOG(INFO) << "After load_model: weights should be loaded";
        text_encoder_->to(options_.device());
        text_encoder_->eval();  // Set text encoder to evaluation mode
        LOG(INFO) << "Qwen2_5_VL text encoder loaded successfully";

        // 验证权重是否被加载
        auto word_embedding =
            text_encoder_->get_language_model()->get_word_embedding();
        if (word_embedding) {
          auto weight = word_embedding->weight();
          LOG(INFO) << "[DEBUG] Word embedding weight shape: "
                    << weight.sizes();
          LOG(INFO) << "[DEBUG] Word embedding weight min/max: "
                    << weight.min().item<float>() << "/"
                    << weight.max().item<float>();
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

  // Prepare text position IDs for text tokens
  torch::Tensor prepare_text_ids(int64_t num_tokens,
                                 int64_t start_height = 0,
                                 int64_t start_width = 0) {
    LOG(INFO) << "[LongCatImage] prepare_text_ids - num_tokens: " << num_tokens
              << ", start_height: " << start_height
              << ", start_width: " << start_width;

    // Create position IDs with correct integer values first
    // Use int64 to ensure precision, then convert to target dtype
    torch::TensorOptions int_options =
        options_.dtype(torch::kInt64).device(options_.device());
    torch::Tensor text_ids_int = torch::zeros({num_tokens, 3}, int_options);

    // modality_id = 0 for text
    text_ids_int.select(1, 0).fill_(0);

    // position indices: [0, 1, 2, ..., num_tokens-1]
    torch::Tensor token_range = torch::arange(num_tokens, int_options);
    text_ids_int.select(1, 1) = token_range + start_height;
    text_ids_int.select(1, 2) = token_range + start_width;

    // Verify integer values before conversion
    LOG(INFO) << "[LongCatImage] prepare_text_ids - token_range shape: "
              << token_range.sizes() << ", dtype: " << token_range.dtype()
              << ", first few: "
              << token_range.slice(0, 0, std::min(5L, num_tokens))
              << ", last few: "
              << token_range.slice(0, std::max(0L, num_tokens - 3), num_tokens)
              << ", min: " << token_range.min().item<int64_t>()
              << ", max: " << token_range.max().item<int64_t>();

    auto last_pos_h_int =
        text_ids_int.select(1, 1).index({num_tokens - 1}).item<int64_t>();
    auto last_pos_w_int =
        text_ids_int.select(1, 2).index({num_tokens - 1}).item<int64_t>();
    LOG(INFO) << "[LongCatImage] prepare_text_ids - last position ID (int64): ("
              << last_pos_h_int << ", " << last_pos_w_int << "), expected: ("
              << (num_tokens - 1 + start_height) << ", "
              << (num_tokens - 1 + start_width) << ")";

    // CRITICAL FIX: Convert to float32 instead of options_.dtype() to avoid
    // precision loss bfloat16 cannot accurately represent 511, causing it to
    // round to 512 Since LongCatImagePosEmbed::forward() converts position IDs
    // to float32 anyway, we should use float32 directly to preserve precision
    torch::TensorOptions float32_options =
        options_.dtype(torch::kFloat32).device(options_.device());
    torch::Tensor text_ids = text_ids_int.to(float32_options);

    // Verify after conversion (read as float to check precision)
    auto last_pos_h =
        text_ids.select(1, 1).index({num_tokens - 1}).item<float>();
    auto last_pos_w =
        text_ids.select(1, 2).index({num_tokens - 1}).item<float>();
    LOG(INFO) << "[LongCatImage] prepare_text_ids - last position ID (after "
                 "conversion to float32): ("
              << last_pos_h << ", " << last_pos_w << "), expected: ("
              << (num_tokens - 1 + start_height) << ", "
              << (num_tokens - 1 + start_width) << ")";

    return text_ids;
  }

  // Prepare image position IDs for latent image tokens
  // Match diffusers: Position IDs should start from 512 (hardcoded
  // tokenizer_max_length) to avoid overlapping with text position IDs
  // Note: Python diffusers uses hardcoded 512, not max_position_embeddings()
  torch::Tensor prepare_latent_image_ids(int64_t batch_size,
                                         int64_t height,
                                         int64_t width) {
    // Match diffusers exactly: use 512 as start_offset
    // This matches diffusers: start=(self.tokenizer_max_length,
    // self.tokenizer_max_length) which is 512
    constexpr int64_t TOKENIZER_MAX_LENGTH = 512;
    int64_t start_offset = TOKENIZER_MAX_LENGTH;  // 512, matching diffusers

    // Create position IDs with correct integer values first (int64), then
    // convert to target dtype This avoids precision loss when adding
    // start_offset
    torch::TensorOptions int_options =
        options_.dtype(torch::kInt64).device(options_.device());
    torch::Tensor latent_image_ids_int =
        torch::zeros({height, width, 3}, int_options);

    // modality_id = 1 for image
    latent_image_ids_int.select(2, 0).fill_(1);

    // Create ranges in int64, add offset, then convert
    torch::Tensor height_range_int =
        torch::arange(height, int_options).unsqueeze(1);
    torch::Tensor width_range_int =
        torch::arange(width, int_options).unsqueeze(0);

    latent_image_ids_int.select(2, 1) += height_range_int + start_offset;
    latent_image_ids_int.select(2, 2) += width_range_int + start_offset;

    // Verify integer values before conversion
    auto first_pos_h_int =
        latent_image_ids_int.select(2, 1).index({0, 0}).item<int64_t>();
    auto first_pos_w_int =
        latent_image_ids_int.select(2, 2).index({0, 0}).item<int64_t>();
    LOG(INFO) << "[LongCatImage] Image position IDs (int64) - start_offset: "
              << start_offset << ", first position ID (int64): ("
              << first_pos_h_int << ", " << first_pos_w_int << "), expected: ("
              << start_offset << ", " << start_offset << ")";

    // CRITICAL FIX: Convert to float32 instead of options_.dtype() to avoid
    // precision loss bfloat16 cannot accurately represent 513, causing it to
    // round to 512 Since LongCatImagePosEmbed::forward() converts position IDs
    // to float32 anyway, we should use float32 directly to preserve precision
    torch::TensorOptions float32_options =
        options_.dtype(torch::kFloat32).device(options_.device());
    torch::Tensor latent_image_ids = latent_image_ids_int.to(float32_options);
    latent_image_ids = latent_image_ids.view({height * width, 3});

    // Verify after conversion
    auto first_pos_h = latent_image_ids.select(1, 1).index({0}).item<float>();
    auto first_pos_w = latent_image_ids.select(1, 2).index({0}).item<float>();
    LOG(INFO) << "[LongCatImage] Image position IDs (after conversion to "
                 "float32) - first position ID: ("
              << first_pos_h << ", " << first_pos_w << "), expected: ("
              << start_offset << ", " << start_offset << ")";
    LOG(INFO) << "[LongCatImage] Image position IDs - first few IDs: "
              << latent_image_ids.slice(0, 0, std::min(3L, height * width));

    return latent_image_ids;
  }

  torch::Tensor pack_latents(const torch::Tensor& latents,
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

  torch::Tensor unpack_latents(const torch::Tensor& latents,
                               int64_t height,
                               int64_t width,
                               int64_t vae_scale_factor) {
    int64_t batch_size = latents.size(0);
    int64_t num_patches = latents.size(1);
    int64_t channels = latents.size(2);

    // Match Python implementation exactly: use integer division
    height = 2 * (height / (vae_scale_factor * 2));
    width = 2 * (width / (vae_scale_factor * 2));

    LOG(INFO) << "[LongCatImage] unpack_latents input shape: "
              << latents.sizes() << ", target height: " << height
              << ", width: " << width
              << ", vae_scale_factor: " << vae_scale_factor;

    torch::Tensor latents_unpacked =
        latents.view({batch_size, height / 2, width / 2, channels / 4, 2, 2});
    latents_unpacked = latents_unpacked.permute({0, 3, 1, 4, 2, 5});
    latents_unpacked = latents_unpacked.reshape(
        {batch_size, channels / (2 * 2), height, width});

    LOG(INFO) << "[LongCatImage] unpack_latents output shape: "
              << latents_unpacked.sizes();

    return latents_unpacked;
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, adjusted_height, adjusted_width};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids = prepare_latent_image_ids(
          batch_size, adjusted_height / 2, adjusted_width / 2);
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor = randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor,
                                                batch_size,
                                                num_channels_latents,
                                                adjusted_height,
                                                adjusted_width);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2);
    return {packed_latents, latent_image_ids};
  }

  // Helper function: split quotation marks (simplified version)
  // For full regex-based implementation, see diffusers implementation
  static std::vector<std::pair<std::string, bool>> split_quotation(
      const std::string& prompt_text) {
    std::vector<std::pair<std::string, bool>> result;
    std::string current;
    bool in_quotes = false;
    char quote_char = '\0';

    for (size_t i = 0; i < prompt_text.length(); ++i) {
      char c = prompt_text[i];
      if ((c == '\'' || c == '\"') && !in_quotes) {
        if (!current.empty()) {
          result.push_back({current, false});
          current.clear();
        }
        in_quotes = true;
        quote_char = c;
        current += c;
      } else if (in_quotes && c == quote_char) {
        current += c;
        result.push_back({current, true});
        current.clear();
        in_quotes = false;
        quote_char = '\0';
      } else {
        current += c;
      }
    }
    if (!current.empty()) {
      result.push_back({current, in_quotes});
    }
    return result;
  }

  // Encode prompt using Qwen2_5_VL text encoder
  torch::Tensor encode_prompt_qwen(std::vector<std::string>& prompt,
                                   int64_t num_images_per_prompt = 1,
                                   int64_t max_sequence_length = 512) {
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded";
    CHECK(!text_encoder_.is_empty()) << "Text encoder not loaded";

    int64_t batch_size = prompt.size();
    std::vector<std::vector<int32_t>> batch_all_tokens;
    batch_all_tokens.reserve(batch_size);

    // Step 1: Tokenize prompts with split_quotation handling
    for (const auto& each_prompt : prompt) {
      std::vector<int32_t> all_tokens;
      auto parts = split_quotation(each_prompt);

      for (const auto& [clean_prompt_sub, matched] : parts) {
        if (matched) {
          // For quoted text, tokenize character by character (simplified)
          // In Python, it's tokenized as-is
          std::vector<int32_t> tokens;
          if (tokenizer_->encode(clean_prompt_sub, &tokens, false)) {
            all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
          }
        } else {
          std::vector<int32_t> tokens;
          if (tokenizer_->encode(clean_prompt_sub, &tokens, false)) {
            all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
          }
        }
      }

      // Log tokenization result for debugging
      if (each_prompt.empty() || each_prompt == "") {
        LOG(INFO) << "[LongCatImage] Empty prompt tokenized to "
                  << all_tokens.size() << " tokens, first few: "
                  << (all_tokens.empty()
                          ? "[]"
                          : (all_tokens.size() <= 5
                                 ? std::to_string(all_tokens[0])
                                 : std::to_string(all_tokens[0]) + ",..."));
      }

      // Truncate if too long
      if (static_cast<int64_t>(all_tokens.size()) > max_sequence_length) {
        LOG(WARNING) << "Input truncated from " << all_tokens.size() << " to "
                     << max_sequence_length << " tokens";
        all_tokens.resize(max_sequence_length);
      }

      batch_all_tokens.push_back(all_tokens);
    }

    // Step 2: Pad tokens to max_sequence_length
    // Find pad_token_id (typically 151643 for Qwen2, but we'll use 0 as
    // default)
    int32_t pad_token_id = 0;
    auto pad_id = tokenizer_->token_to_id("<|endoftext|>");
    if (pad_id.has_value()) {
      pad_token_id = pad_id.value();
    }

    // Pad each sequence
    for (auto& tokens : batch_all_tokens) {
      int64_t pad_len =
          max_sequence_length - static_cast<int64_t>(tokens.size());
      if (pad_len > 0) {
        tokens.insert(tokens.end(), pad_len, pad_token_id);
      }
    }

    // Step 3: Tokenize prefix and suffix templates
    std::vector<int32_t> prefix_tokens;
    CHECK(tokenizer_->encode(
        prompt_template_encode_prefix_, &prefix_tokens, false));
    std::vector<int32_t> suffix_tokens;
    CHECK(tokenizer_->encode(
        prompt_template_encode_suffix_, &suffix_tokens, false));
    int64_t prefix_len = prefix_tokens.size();
    int64_t suffix_len = suffix_tokens.size();

    // Step 4: Concatenate prefix + tokens + suffix
    int64_t total_seq_len = prefix_len + max_sequence_length + suffix_len;
    std::vector<int32_t> input_ids_flat;
    std::vector<int32_t> attention_mask_flat;
    input_ids_flat.reserve(batch_size * total_seq_len);
    attention_mask_flat.reserve(batch_size * total_seq_len);

    for (int64_t i = 0; i < batch_size; ++i) {
      // Add prefix
      input_ids_flat.insert(
          input_ids_flat.end(), prefix_tokens.begin(), prefix_tokens.end());
      attention_mask_flat.insert(attention_mask_flat.end(), prefix_len, 1);

      // Add tokens
      input_ids_flat.insert(input_ids_flat.end(),
                            batch_all_tokens[i].begin(),
                            batch_all_tokens[i].end());
      // Count non-padding tokens for attention mask
      int64_t non_pad_count = 0;
      for (const auto& token : batch_all_tokens[i]) {
        if (token != pad_token_id) {
          non_pad_count++;
        } else {
          break;  // Padding starts here
        }
      }
      // Mark real tokens as 1, padding as 0
      for (int64_t j = 0; j < max_sequence_length; ++j) {
        attention_mask_flat.push_back(j < non_pad_count ? 1 : 0);
      }

      // Add suffix
      input_ids_flat.insert(
          input_ids_flat.end(), suffix_tokens.begin(), suffix_tokens.end());
      attention_mask_flat.insert(attention_mask_flat.end(), suffix_len, 1);
    }

    // Create input_ids tensor [batch_size, total_seq_len]
    torch::Tensor input_ids =
        torch::tensor(input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(options_.device());

    // Create attention_mask tensor [batch_size, total_seq_len]
    torch::Tensor attention_mask =
        torch::tensor(attention_mask_flat, torch::dtype(torch::kLong))
            .view({batch_size, total_seq_len})
            .to(options_.device());

    LOG(INFO) << "[LongCatImage] attention_mask shape: "
              << attention_mask.sizes() << ", dtype: " << attention_mask.dtype()
              << ", first few values (first batch): "
              << attention_mask.slice(0, 0, 1).slice(
                     1, 0, std::min(20L, total_seq_len));

    // Step 5: Get hidden states from text encoder
    // Following diffusers implementation exactly
    // The key is that attention_mask is used to create an additive mask for the
    // attention mechanism This is applied INSIDE the transformer layers, not
    // post-hoc

    ModelInputParams input_params;

    // Flatten input_ids for forward pass: [batch_size, total_seq_len] ->
    // [batch_size * total_seq_len]
    torch::Tensor tokens_flat = input_ids.view({-1});

    // Create MROPE-style positions tensor: [3, seq_len]
    // HuggingFace Qwen2_5_VL uses MROPE (Multimodal RoPE) even for text-only:
    //   position_ids = attention_mask.long().cumsum(-1) - 1
    //   position_ids.masked_fill_(attention_mask == 0, 1)
    //   position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    // For text-only, all 3 MROPE dimensions (T, H, W) use the same positions
    auto mask_flat = attention_mask.view({-1});  // [batch * seq_len]
    auto positions_1d = mask_flat.to(torch::kInt64).cumsum(-1) - 1;
    positions_1d = positions_1d.masked_fill(mask_flat == 0, 1);

    // Expand to [3, seq_len] for MROPE - this triggers xllm's apply_mrope()
    // in llm_model_base.h when positions.dim() == 2
    torch::Tensor positions_2d =
        positions_1d.unsqueeze(0).expand({3, -1}).contiguous();

    LOG(INFO) << "[LongCatImage] MROPE positions - shape: "
              << positions_2d.sizes() << ", positions_1d sample (first 10): "
              << positions_1d.slice(0, 0, 10)
              << ", positions_1d sample (around padding start "
              << prefix_len - 5 << "-" << prefix_len + 5
              << "): " << positions_1d.slice(0, prefix_len - 5, prefix_len + 5);

    // Prepare empty kv_caches for forward pass
    std::vector<KVCache> kv_caches;

    LOG(INFO) << "[LongCatImage] ===== Starting Text Encoder Forward Pass ====="
              << "\n[LongCatImage] input_ids shape: " << input_ids.sizes()
              << ", dtype: " << input_ids.dtype()
              << "\n[LongCatImage] attention_mask shape: "
              << attention_mask.sizes() << ", dtype: " << attention_mask.dtype()
              << "\n[LongCatImage] Prefix length: " << prefix_len
              << ", Suffix length: " << suffix_len
              << ", Total sequence length: " << total_seq_len;

    // Print input parameters for forward_longcat call (for diffusers
    // comparison)
    LOG(INFO) << "[LongCatImage] ===== forward_longcat Input Parameters ====="
              << "\n[LongCatImage] tokens_flat shape: " << tokens_flat.sizes()
              << ", dtype: " << tokens_flat.dtype()
              << ", device: " << tokens_flat.device()
              << "\n[LongCatImage] tokens_flat values: " << tokens_flat
              << "\n[LongCatImage] positions_2d (MROPE) shape: "
              << positions_2d.sizes() << ", dtype: " << positions_2d.dtype()
              << ", device: " << positions_2d.device()
              << "\n[LongCatImage] kv_caches size: " << kv_caches.size()
              << "\n[LongCatImage] attention_mask shape: "
              << attention_mask.sizes() << ", dtype: " << attention_mask.dtype()
              << ", device: " << attention_mask.device()
              << "\n[LongCatImage] attention_mask values: " << attention_mask;

    // Call text encoder forward pass with attention_mask using the
    // LongCat-specific method. Using 2D positions [3, seq_len] to enable MROPE
    // which matches HuggingFace's Qwen2_5_VL behavior exactly
    torch::Tensor hidden_states_flat = text_encoder_->forward_longcat(
        tokens_flat, positions_2d, kv_caches, input_params, attention_mask);

    // Reshape: [batch_size * total_seq_len, hidden_size] -> [batch_size,
    // total_seq_len, hidden_size]
    int64_t hidden_size = hidden_states_flat.size(-1);
    torch::Tensor hidden_states_last =
        hidden_states_flat.view({batch_size, total_seq_len, hidden_size});

    LOG(INFO) << "[LongCatImage] Text encoder forward output - shape: "
              << hidden_states_last.sizes() << "\n[LongCatImage] Stats: min: "
              << hidden_states_last.min().item<float>()
              << ", max: " << hidden_states_last.max().item<float>()
              << ", mean: " << hidden_states_last.mean().item<float>()
              << ", std: " << hidden_states_last.std().item<float>();

    // Step 6: Remove prefix and suffix tokens
    // This matches diffusers: prompt_embeds =
    // text_output.hidden_states[-1].detach()
    //                          prompt_embeds = prompt_embeds[:,
    //                          prefix_len:-suffix_len, :]
    // Result shape: [batch_size, max_sequence_length, hidden_size]
    torch::Tensor prompt_embeds =
        hidden_states_last.slice(1, prefix_len, total_seq_len - suffix_len);

    LOG(INFO) << "[LongCatImage] After removing prefix/suffix - shape: "
              << prompt_embeds.sizes() << "\n[LongCatImage] Stats: min: "
              << prompt_embeds.min().item<float>()
              << ", max: " << prompt_embeds.max().item<float>()
              << ", mean: " << prompt_embeds.mean().item<float>()
              << ", std: " << prompt_embeds.std().item<float>();

    // Step 7: Repeat for num_images_per_prompt (matches diffusers
    // encode_prompt) duplicate text embeddings for each generation per prompt
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    // Reshape: [batch_size, max_sequence_length * num_images_per_prompt,
    // hidden_size]
    //       -> [batch_size * num_images_per_prompt, max_sequence_length,
    //       hidden_size]
    prompt_embeds = prompt_embeds.view(
        {batch_size * num_images_per_prompt, max_sequence_length, -1});

    LOG(INFO) << "[LongCatImage] ===== encode_prompt_qwen Final Result ====="
              << "\n[LongCatImage] Final shape: " << prompt_embeds.sizes()
              << "\n[LongCatImage] Final stats: min: "
              << prompt_embeds.min().item<float>()
              << ", max: " << prompt_embeds.max().item<float>()
              << ", mean: " << prompt_embeds.mean().item<float>()
              << ", std: " << prompt_embeds.std().item<float>();

    return prompt_embeds.to(options_);
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<torch::Tensor> prompt_embeds,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }

    if (!prompt_embeds.has_value()) {
      prompt_embeds = encode_prompt_qwen(
          prompt_list, num_images_per_prompt, max_sequence_length);
    }

    int64_t seq_len = prompt_embeds.value().size(1);
    LOG(INFO) << "[LongCatImage] encode_prompt - seq_len: " << seq_len
              << ", prompt_embeds shape: " << prompt_embeds.value().sizes();
    torch::Tensor text_ids = prepare_text_ids(seq_len);

    LOG(INFO) << "[LongCatImage] Text position IDs - shape: "
              << text_ids.sizes() << ", first few IDs: "
              << text_ids.slice(0, 0, std::min(5L, seq_len))
              << ", last few IDs: "
              << text_ids.slice(0, std::max(0L, seq_len - 3), seq_len);

    LOG(INFO) << "[LongCatImage] encode_prompt result shape: "
              << prompt_embeds.value().sizes();
    LOG(INFO) << "[LongCatImage] encode_prompt min: "
              << prompt_embeds.value().min().item<float>()
              << ", max: " << prompt_embeds.value().max().item<float>()
              << ", mean: " << prompt_embeds.value().mean().item<float>()
              << ", std: " << prompt_embeds.value().std().item<float>();

    return {prompt_embeds.value(), text_ids};
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
  std::string prompt_template_encode_prefix_;
  std::string prompt_template_encode_suffix_;

  std::vector<torch::Tensor> forward_(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt_2 = std::nullopt,
      float true_cfg_scale = 1.0f,
      int64_t height = 768,
      int64_t width = 1344,
      int64_t num_inference_steps = 50,
      float guidance_scale = 4.5f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_pooled_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512,
      bool enable_cfg_renorm = true,
      float cfg_renorm_min = 0.0f) {
    torch::NoGradGuard no_grad;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0) / num_images_per_prompt;
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;

    LOG(INFO) << "[LongCatImage] CFG debug - guidance_scale: " << guidance_scale
              << ", guidance_scale > 1.0f: " << (guidance_scale > 1.0f)
              << ", negative_prompt.has_value(): "
              << negative_prompt.has_value()
              << ", negative_prompt_embeds.has_value(): "
              << negative_prompt_embeds.has_value();

    // Enable CFG when guidance_scale > 1.0, regardless of negative_prompt
    // availability If no negative_prompt is provided, we'll use an empty string
    bool do_classifier_free_guidance = (guidance_scale > 1.0f);

    // If CFG is enabled but no negative_prompt provided, use empty string
    if (do_classifier_free_guidance && !negative_prompt.has_value() &&
        !negative_prompt_embeds.has_value()) {
      LOG(INFO) << "[LongCatImage] CFG enabled but no negative_prompt "
                   "provided, using empty string";
      negative_prompt = std::vector<std::string>{""};
    }

    LOG(INFO) << "[LongCatImage] CFG config - do_classifier_free_guidance: "
              << do_classifier_free_guidance;

    // Encode prompt
    auto [encoded_prompt_embeds, text_ids] = encode_prompt(
        prompt, prompt_embeds, num_images_per_prompt, max_sequence_length);

    // Encode negative prompt
    torch::Tensor negative_encoded_embeds;
    torch::Tensor negative_text_ids;
    if (do_classifier_free_guidance) {
      auto [neg_emb, neg_ids] = encode_prompt(negative_prompt,
                                              negative_prompt_embeds,
                                              num_images_per_prompt,
                                              max_sequence_length);
      negative_encoded_embeds = neg_emb;
      negative_text_ids = neg_ids;

      LOG(INFO) << "[LongCatImage] Negative prompt encoded - shape: "
                << negative_encoded_embeds.sizes()
                << ", min: " << negative_encoded_embeds.min().item<float>()
                << ", max: " << negative_encoded_embeds.max().item<float>()
                << ", mean: " << negative_encoded_embeds.mean().item<float>();
      LOG(INFO) << "[LongCatImage] Positive prompt encoded - shape: "
                << encoded_prompt_embeds.sizes()
                << ", min: " << encoded_prompt_embeds.min().item<float>()
                << ", max: " << encoded_prompt_embeds.max().item<float>()
                << ", mean: " << encoded_prompt_embeds.mean().item<float>();
    }

    // Prepare latent variables
    int64_t num_channels_latents = 16;  // LongCat-Image uses 16 channels
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.has_value() ? seed.value() : 42,
                        latents);

    LOG(INFO) << "[LongCatImage] prepared_latents shape: "
              << prepared_latents.sizes();
    LOG(INFO) << "[LongCatImage] prepared_latents min: "
              << prepared_latents.min().item<float>()
              << ", max: " << prepared_latents.max().item<float>()
              << ", mean: " << prepared_latents.mean().item<float>()
              << ", std: " << prepared_latents.std().item<float>();

    // Prepare timesteps
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

    LOG(INFO) << "[LongCatImage] timesteps shape: " << timesteps.sizes();
    LOG(INFO) << "[LongCatImage] timesteps values: ["
              << timesteps[0].item<float>() << ", "
              << timesteps[1].item<float>() << ", ..., "
              << timesteps[-1].item<float>() << "]";

    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());

    // Image rotary positional embeddings
    LOG(INFO) << "[LongCatImage] Before forward_cache - text_ids shape: "
              << text_ids.sizes()
              << ", latent_image_ids shape: " << latent_image_ids.sizes();
    LOG(INFO) << "[LongCatImage] text_ids last few: "
              << text_ids.slice(
                     0, std::max(0L, text_ids.size(0) - 3), text_ids.size(0));
    LOG(INFO) << "[LongCatImage] latent_image_ids first few: "
              << latent_image_ids.slice(
                     0, 0, std::min(3L, latent_image_ids.size(0)));

    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height / (vae_scale_factor_ * 2),
                                  width / (vae_scale_factor_ * 2));
    std::vector<torch::Tensor> rot_embs = {rot_emb1, rot_emb2};
    torch::Tensor image_rotary_emb = torch::stack(rot_embs, 0);

    // Denoising loop
    LOG(INFO)
        << "[LongCatImage] Before denoising loop - prepared_latents shape: "
        << prepared_latents.sizes()
        << ", encoded_prompt_embeds shape: " << encoded_prompt_embeds.sizes()
        << ", image_rotary_emb shape: " << image_rotary_emb.sizes();

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);

      // Forward through transformer
      if (i == 0) {
        LOG(INFO)
            << "[LongCatImage] Step 0 - Transformer inputs - latents shape: "
            << prepared_latents.sizes()
            << ", prompt_embeds shape: " << encoded_prompt_embeds.sizes()
            << ", timestep: " << timestep.item<float>()
            << ", rotary_emb shape: " << image_rotary_emb.sizes();
      }
      torch::Tensor noise_pred = transformer_->forward(
          prepared_latents, encoded_prompt_embeds, timestep, image_rotary_emb);

      LOG(INFO) << "[LongCatImage] Step " << i << "/" << timesteps.numel()
                << " - noise_pred shape: " << noise_pred.sizes();
      LOG(INFO) << "[LongCatImage] Step " << i
                << " - noise_pred min: " << noise_pred.min().item<float>()
                << ", max: " << noise_pred.max().item<float>()
                << ", mean: " << noise_pred.mean().item<float>()
                << ", std: " << noise_pred.std().item<float>();

      if (do_classifier_free_guidance) {
        // Forward negative prompt
        // Use a different step_idx (i + 10000) to avoid cache collision
        // This ensures positive and negative prompts use separate cache entries
        torch::Tensor negative_noise_pred = transformer_->forward(
            prepared_latents,
            negative_encoded_embeds,
            timestep,
            image_rotary_emb,
            i + 10000);  // Use different step_idx to avoid cache collision

        // Save conditional prediction before CFG
        torch::Tensor noise_pred_text = noise_pred;

        LOG(INFO) << "[LongCatImage] Step " << i
                  << " - Before CFG - noise_pred_text min: "
                  << noise_pred_text.min().item<float>()
                  << ", max: " << noise_pred_text.max().item<float>()
                  << ", mean: " << noise_pred_text.mean().item<float>();
        LOG(INFO) << "[LongCatImage] Step " << i
                  << " - Before CFG - negative_noise_pred min: "
                  << negative_noise_pred.min().item<float>()
                  << ", max: " << negative_noise_pred.max().item<float>()
                  << ", mean: " << negative_noise_pred.mean().item<float>();

        // Classifier-free guidance
        torch::Tensor cfg_noise_pred_before = noise_pred;
        noise_pred = negative_noise_pred +
                     guidance_scale * (noise_pred_text - negative_noise_pred);

        LOG(INFO) << "[LongCatImage] Step " << i
                  << " - CFG applied, guidance_scale: " << guidance_scale;
        LOG(INFO) << "[LongCatImage] Step " << i
                  << " - noise_pred after CFG min: "
                  << noise_pred.min().item<float>()
                  << ", max: " << noise_pred.max().item<float>()
                  << ", mean: " << noise_pred.mean().item<float>();

        // CFG renorm
        if (enable_cfg_renorm) {
          torch::Tensor cond_norm = torch::norm(noise_pred_text, 2, -1, true);
          torch::Tensor noise_norm = torch::norm(noise_pred, 2, -1, true);
          torch::Tensor scale = (cond_norm / (noise_norm + 1e-8f))
                                    .clamp_min(cfg_renorm_min)
                                    .clamp_max(1.0f);
          noise_pred = noise_pred * scale;

          LOG(INFO) << "[LongCatImage] Step " << i
                    << " - CFG renorm applied, scale min: "
                    << scale.min().item<float>()
                    << ", max: " << scale.max().item<float>();
        }
      }

      // Scheduler step
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();

      LOG(INFO) << "[LongCatImage] Step " << i
                << " - after scheduler step, latents min: "
                << prepared_latents.min().item<float>()
                << ", max: " << prepared_latents.max().item<float>()
                << ", mean: " << prepared_latents.mean().item<float>()
                << ", std: " << prepared_latents.std().item<float>();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    // Decode latents to images
    torch::Tensor image;
    torch::Tensor unpacked_latents =
        unpack_latents(prepared_latents, height, width, vae_scale_factor_);

    LOG(INFO) << "[LongCatImage] unpacked_latents shape: "
              << unpacked_latents.sizes();
    LOG(INFO) << "[LongCatImage] unpacked_latents min: "
              << unpacked_latents.min().item<float>()
              << ", max: " << unpacked_latents.max().item<float>()
              << ", mean: " << unpacked_latents.mean().item<float>()
              << ", std: " << unpacked_latents.std().item<float>();

    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;

    LOG(INFO) << "[LongCatImage] after VAE scaling/shift, latents min: "
              << unpacked_latents.min().item<float>()
              << ", max: " << unpacked_latents.max().item<float>()
              << ", mean: " << unpacked_latents.mean().item<float>();

    // Ensure VAE input matches model dtype (VAE is loaded in bfloat16)
    unpacked_latents = unpacked_latents.to(options_.dtype());

    LOG(INFO) << "[LongCatImage] VAE input latents min: "
              << unpacked_latents.min().item<float>()
              << ", max: " << unpacked_latents.max().item<float>()
              << ", mean: " << unpacked_latents.mean().item<float>()
              << ", std: " << unpacked_latents.std().item<float>();

    LOG(INFO) << "[LongCatImage] Calling VAE decode with input shape: "
              << unpacked_latents.sizes();
    if (!vae_) {
      LOG(ERROR) << "[LongCatImage] VAE is null!";
      return std::vector<torch::Tensor>{
          {torch::zeros({1, 3, height, width}, options_)}};
    }

    try {
      image = vae_->decode(unpacked_latents);
      LOG(INFO) << "[LongCatImage] VAE decode completed, output shape: "
                << image.sizes();
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LongCatImage] VAE decode failed with exception: "
                 << e.what();
      return std::vector<torch::Tensor>{
          {torch::zeros({1, 3, height, width}, options_)}};
    } catch (...) {
      LOG(ERROR) << "[LongCatImage] VAE decode failed with unknown exception";
      return std::vector<torch::Tensor>{
          {torch::zeros({1, 3, height, width}, options_)}};
    }

    LOG(INFO) << "[LongCatImage] VAE decoded image shape: " << image.sizes();
    LOG(INFO) << "[LongCatImage] VAE decoded image min: "
              << image.min().item<float>()
              << ", max: " << image.max().item<float>()
              << ", mean: " << image.mean().item<float>()
              << ", std: " << image.std().item<float>();

    image = vae_image_processor_->postprocess(image);

    LOG(INFO) << "[LongCatImage] After VAE postprocess min: "
              << image.min().item<float>()
              << ", max: " << image.max().item<float>()
              << ", mean: " << image.mean().item<float>()
              << ", std: " << image.std().item<float>();

    LOG(INFO) << "[LongCatImage] Final image shape: " << image.sizes();
    LOG(INFO) << "[LongCatImage] Final image min: " << image.min().item<float>()
              << ", max: " << image.max().item<float>()
              << ", mean: " << image.mean().item<float>();

    // Ensure image is on CPU and in float32 format for encoding
    image = image.cpu().to(torch::kFloat32).contiguous();

    LOG(INFO) << "[LongCatImage] Final image device: " << image.device()
              << ", dtype: " << image.dtype();

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
