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
#include "core/layers/pos_embedding.h"
#include "dit.h"
#include "pipeline_flux_base.h"

// LongCat-Image pipeline implementation
// Uses Qwen2_5_VL as text encoder instead of CLIP+T5

namespace xllm {

class LongCatImagePipelineImpl : public FluxPipelineBaseImpl {
 public:
  LongCatImagePipelineImpl(const DiTModelContext& context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);

    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    tokenizer_max_length_ =
        context.get_model_args("text_encoder").max_position_embeddings();
    LOG(INFO) << "Initializing LongCat-Image pipeline...";
    vae_image_processor_ = VAEImageProcessor(context.get_model_context("vae"),
                                             true,
                                             true,
                                             false,
                                             false,
                                             false,
                                             model_args.latent_channels());
    vae_ = VAE(context.get_model_context("vae"));
    pos_embed_ = register_module(
        "pos_embed",
        FluxPosEmbed(ROPE_SCALE_BASE,
                     context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = FluxDiTModel(context.get_model_context("transformer"));

    // LongCat-Image uses Qwen2_5_VL as text encoder, not CLIP+T5
    // For now, we load it as text_encoder
    // TODO: Integrate VLM model for text encoding
    LOG(INFO) << "LongCat-Image uses Qwen2_5_VL as text encoder";

    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));
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
    std::string model_path = loader->model_root_path();
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

    // TODO: Load VLM text encoder model
    // For now, we skip loading the text encoder as it requires VLM integration
    LOG(INFO) << "Text encoder (Qwen2_5_VL) loading is not yet implemented";

    // Load tokenizer if available
    if (tokenizer_loader) {
      tokenizer_ = tokenizer_loader->tokenizer();
      LOG(INFO) << "Tokenizer loaded successfully";
    }
  }

 private:
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

    // TODO: Encode prompt using VLM text encoder
    // For now, use dummy embeddings
    LOG(WARNING)
        << "Using dummy text embeddings - VLM text encoder not implemented";
    torch::Tensor encoded_prompt_embeds;
    torch::Tensor encoded_pooled_embeds;
    torch::Tensor text_ids;

    if (prompt_embeds.has_value()) {
      encoded_prompt_embeds = prompt_embeds.value();
      encoded_pooled_embeds = pooled_prompt_embeds.value();
    } else {
      // Create dummy embeddings for testing
      int64_t hidden_size = transformer_->joint_attention_dim();
      encoded_prompt_embeds = torch::zeros(
          {total_batch_size, max_sequence_length, hidden_size}, options_);
      encoded_pooled_embeds = torch::zeros(
          {total_batch_size, transformer_->pooled_projection_dim()}, options_);
    }

    text_ids =
        torch::zeros({total_batch_size, max_sequence_length, 3},
                     torch::dtype(torch::kLong).device(options_.device()));

    // encode negative prompt
    torch::Tensor negative_encoded_embeds, negative_pooled_embeds;
    torch::Tensor negative_text_ids;
    if (do_true_cfg) {
      if (negative_prompt_embeds.has_value()) {
        negative_encoded_embeds = negative_prompt_embeds.value();
        negative_pooled_embeds = negative_pooled_prompt_embeds.value();
      } else {
        int64_t hidden_size = transformer_->joint_attention_dim();
        negative_encoded_embeds = torch::zeros(
            {total_batch_size, max_sequence_length, hidden_size}, options_);
        negative_pooled_embeds = torch::zeros(
            {total_batch_size, transformer_->pooled_projection_dim()},
            options_);
      }
      negative_text_ids =
          torch::zeros({total_batch_size, max_sequence_length, 3},
                       torch::dtype(torch::kLong).device(options_.device()));
    }

    // prepare latent
    int64_t num_channels_latents = transformer_->in_channels() / 4;
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        seed.has_value() ? seed.value() : 42,
                        latents);

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
      torch::Tensor noise_pred = transformer_->forward(prepared_latents,
                                                       encoded_prompt_embeds,
                                                       encoded_pooled_embeds,
                                                       timestep,
                                                       image_rotary_emb,
                                                       guidance,
                                                       step_id);
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
      prepared_latents = prev_latents.detach();
      std::vector<torch::Tensor> tensors = {prepared_latents, noise_pred};
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }
    torch::Tensor image;
    // Unpack latents
    torch::Tensor unpacked_latents =
        unpack_latents(prepared_latents, height, width, vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
    unpacked_latents = unpacked_latents.to(options_.dtype());
    image = vae_->decode(unpacked_latents);
    image = vae_image_processor_->postprocess(image);
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