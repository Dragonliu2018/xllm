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

// MiMo-V2.5-ASR: Speech Language Model for CUDA inference.
//
// ──────────────────────────────────────────────────────────────────────────────
// SLM pattern for xLLM  (future models follow the same three-step interface)
// ──────────────────────────────────────────────────────────────────────────────
//
//  1. Processor  (processors/mimo_audio_processor.h)
//     raw waveform / pre-computed RVQ codes
//     → AudioProcessor::process() stores in mm_data["audio|codes"]
//
//  2. Model::get_multimodal_embeddings()          [prefill only]
//     mm_data["audio|codes"]  [T, audio_channels]
//     → MiMoSpeechEncoder::encode()               speech-group embeddings
//     → returns MMDict{"audio|embedding": [T_groups, hidden_size]}
//
//  3. Model::get_input_embeddings()               [prefill only]
//     text embeddings  +  mm_data["audio|embedding"]
//     → replace <|empty|> positions with speech-group embeddings
//     → returns merged [num_tokens, hidden_size]
//
//  4. Model::forward()  (inherited from LlmModelImplBase)
//     uses params.embedding.input_embedding (set by serving framework)
//     → standard Qwen2 global transformer
//     → returns ModelOutput{hidden_states}
//
//  5. Model::logits()
//     → text LM head → [num_selected, vocab_size]
//
//  6. (TTS only) MiMoLocalTransformer::local_forward()
//     global hidden state  →  [group_size, audio_channels] speech tokens
//
// Reference: MiMo-V2.5-ASR/src/mimo_audio/modeling_mimo_audio.py

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/qwen2_decoder_layer.h"
#include "models/llm/llm_model_base.h"
#include "models/model_registry.h"
#include "processors/multimodal_processor.h"

namespace xllm {

// ─── Anonymous-namespace helpers ─────────────────────────────────────────────
// Used only by the local / speech-group transformers, which run SDPA on very
// short sequences (≤ group_size + max_delay ≈ 11 tokens) and therefore do not
// need paged KV-caches or FlashInfer.

namespace {

// Half-rotate: RoPE interleave used by Qwen2.
// x: [B, H, T, D],  cos/sin: [1, 1, T, D]
torch::Tensor apply_rotary_half(const torch::Tensor& x,
                                const torch::Tensor& cos,
                                const torch::Tensor& sin) {
  const int64_t d = x.size(-1);
  auto x1 = x.slice(-1, 0, d / 2);
  auto x2 = x.slice(-1, d / 2, d);
  return x * cos + torch::cat({-x2, x1}, -1) * sin;
}

}  // namespace

// ─── MiMoSimpleRMSNorm ───────────────────────────────────────────────────────

class MiMoSimpleRMSNormImpl final : public torch::nn::Module {
 public:
  MiMoSimpleRMSNormImpl(int64_t dim, float eps = 1e-6f) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({dim}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    auto fp = x.to(torch::kFloat32);
    fp = fp * torch::rsqrt(fp.pow(2).mean(-1, /*keepdim=*/true) + eps_);
    return weight_.to(x.dtype()) * fp.to(x.dtype());
  }

  void load_state_dict(const StateDict& sd) {
    auto w = sd.get_tensor("weight");
    if (w.defined()) {
      weight_.data().copy_(w);
    }
  }

 private:
  float eps_;
  torch::Tensor weight_;
};
TORCH_MODULE(MiMoSimpleRMSNorm);

// ─── MiMoSiluMLP ─────────────────────────────────────────────────────────────
// SwiGLU: down_proj( silu(gate) ⊙ up )

class MiMoSiluMLPImpl final : public torch::nn::Module {
 public:
  MiMoSiluMLPImpl(int64_t hidden, int64_t ffn) {
    gate_ = register_module(
        "gate_proj",
        torch::nn::Linear(torch::nn::LinearOptions(hidden, ffn).bias(false)));
    up_ = register_module(
        "up_proj",
        torch::nn::Linear(torch::nn::LinearOptions(hidden, ffn).bias(false)));
    down_ = register_module(
        "down_proj",
        torch::nn::Linear(torch::nn::LinearOptions(ffn, hidden).bias(false)));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return down_->forward(torch::nn::functional::silu(gate_->forward(x)) *
                          up_->forward(x));
  }

  void load_state_dict(const StateDict& sd) {
    auto load = [&](const std::string& k, torch::nn::Linear& m) {
      auto w = sd.get_tensor(k + ".weight");
      if (w.defined()) {
        m->weight.data().copy_(w);
      }
    };
    load("gate_proj", gate_);
    load("up_proj", up_);
    load("down_proj", down_);
  }

 private:
  torch::nn::Linear gate_{nullptr}, up_{nullptr}, down_{nullptr};
};
TORCH_MODULE(MiMoSiluMLP);

// ─── MiMoSdpaAttention ───────────────────────────────────────────────────────
// Multi-head attention with incremental KV-cache (for local/speech-group
// transformers).  Uses torch::nn::functional::scaled_dot_product_attention,
// which dispatches to Flash-Attention automatically on CUDA.

class MiMoSdpaAttentionImpl final : public torch::nn::Module {
 public:
  MiMoSdpaAttentionImpl(int64_t hidden,
                        int32_t n_heads,
                        int64_t max_seq,
                        float rope_theta = 10000.0f)
      : n_heads_(n_heads), head_dim_(hidden / n_heads) {
    q_proj_ = register_module(
        "q_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
    k_proj_ = register_module(
        "k_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
    v_proj_ = register_module(
        "v_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));
    o_proj_ = register_module(
        "o_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden, hidden).bias(false)));

    // Pre-compute RoPE tables
    auto idx = torch::arange(0, head_dim_, 2, torch::kFloat32);
    inv_freq_ = register_buffer(
        "inv_freq",
        1.0f / torch::pow(rope_theta, idx / static_cast<float>(head_dim_)));
    auto pos = torch::arange(max_seq, torch::kFloat32);
    auto freqs = torch::outer(pos, inv_freq_);
    auto emb = torch::cat({freqs, freqs}, -1);
    cos_table_ = register_buffer("cos_table", emb.cos());
    sin_table_ = register_buffer("sin_table", emb.sin());
  }

  // x: [B, T, H]    past_k/v: optional [B, n_heads, T_past, D]
  // Returns (output, new_k, new_v).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& past_k,
      const torch::Tensor& past_v,
      bool is_causal) {
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t offset = past_k.defined() ? past_k.size(2) : 0;

    auto reshape = [&](const torch::Tensor& t) {
      return t.view({B, T, n_heads_, head_dim_}).transpose(1, 2);
    };
    auto q = reshape(q_proj_->forward(x));
    auto k = reshape(k_proj_->forward(x));
    auto v = reshape(v_proj_->forward(x));

    // Apply RoPE at the correct absolute position
    auto cos =
        cos_table_.slice(0, offset, offset + T).unsqueeze(0).unsqueeze(0);
    auto sin =
        sin_table_.slice(0, offset, offset + T).unsqueeze(0).unsqueeze(0);
    q = apply_rotary_half(q, cos, sin);
    k = apply_rotary_half(k, cos, sin);

    // Append to incremental KV-cache
    auto new_k = past_k.defined() ? torch::cat({past_k, k}, 2) : k;
    auto new_v = past_v.defined() ? torch::cat({past_v, v}, 2) : v;

    // SDPA.  Use is_causal only when q_len == kv_len (full-sequence prefill).
    // For incremental decode (T=1 with cache), skip causal mask—the cache
    // already encodes causality.
    const bool use_causal = is_causal && (T == new_k.size(2));
    auto attn =
        torch::scaled_dot_product_attention(q,
                                            new_k,
                                            new_v,
                                            torch::nullopt,  // attn_mask
                                            0.0,             // dropout_p
                                            use_causal       // is_causal
        );

    auto out =
        attn.transpose(1, 2).contiguous().view({B, T, n_heads_ * head_dim_});
    return {o_proj_->forward(out), new_k, new_v};
  }

  void load_state_dict(const StateDict& sd) {
    auto load = [&](const std::string& k, torch::nn::Linear& m) {
      auto w = sd.get_tensor(k + ".weight");
      if (w.defined()) {
        m->weight.data().copy_(w);
      }
    };
    load("q_proj", q_proj_);
    load("k_proj", k_proj_);
    load("v_proj", v_proj_);
    load("o_proj", o_proj_);
  }

 private:
  int32_t n_heads_;
  int64_t head_dim_;
  torch::nn::Linear q_proj_{nullptr}, k_proj_{nullptr}, v_proj_{nullptr},
      o_proj_{nullptr};
  torch::Tensor inv_freq_, cos_table_, sin_table_;
};
TORCH_MODULE(MiMoSdpaAttention);

// ─── MiMoSdpaDecoderLayer ────────────────────────────────────────────────────

class MiMoSdpaDecoderLayerImpl final : public torch::nn::Module {
 public:
  MiMoSdpaDecoderLayerImpl(int64_t hidden,
                           int32_t n_heads,
                           int64_t ffn,
                           int64_t max_seq,
                           float eps = 1e-6f,
                           float rope_theta = 10000.0f) {
    input_norm_ =
        register_module("input_layernorm", MiMoSimpleRMSNorm(hidden, eps));
    attn_ = register_module(
        "self_attn", MiMoSdpaAttention(hidden, n_heads, max_seq, rope_theta));
    post_norm_ = register_module("post_attention_layernorm",
                                 MiMoSimpleRMSNorm(hidden, eps));
    mlp_ = register_module("mlp", MiMoSiluMLP(hidden, ffn));
  }

  // Returns (output, new_k, new_v)
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& past_k,
      const torch::Tensor& past_v,
      bool is_causal) {
    auto normed = input_norm_->forward(x);
    auto [attn_out, new_k, new_v] =
        attn_->forward(normed, past_k, past_v, is_causal);
    auto h = x + attn_out;
    h = h + mlp_->forward(post_norm_->forward(h));
    return {h, new_k, new_v};
  }

  void load_state_dict(const StateDict& sd) {
    input_norm_->load_state_dict(sd.get_dict_with_prefix("input_layernorm."));
    attn_->load_state_dict(sd.get_dict_with_prefix("self_attn."));
    post_norm_->load_state_dict(
        sd.get_dict_with_prefix("post_attention_layernorm."));
    mlp_->load_state_dict(sd.get_dict_with_prefix("mlp."));
  }

 private:
  MiMoSimpleRMSNorm input_norm_{nullptr}, post_norm_{nullptr};
  MiMoSdpaAttention attn_{nullptr};
  MiMoSiluMLP mlp_{nullptr};
};
TORCH_MODULE(MiMoSdpaDecoderLayer);

// ─── MiMoSpeechGroupEncoderImpl ──────────────────────────────────────────────
// Encodes speech groups independently.
// Input:  [T_groups, group_size, embed_dim]   (each group is a mini-batch)
// Output: [T_groups, group_size, embed_dim]
//
// Maps to: input_local_transformer in the Python model.

class MiMoSpeechGroupEncoderImpl final : public torch::nn::Module {
 public:
  MiMoSpeechGroupEncoderImpl(int32_t n_layers,
                             int64_t hidden,
                             int32_t n_heads,
                             int64_t ffn,
                             int64_t group_size,
                             bool full_attention = false,
                             float eps = 1e-6f,
                             float rope_theta = 10000.0f)
      : is_causal_(!full_attention) {
    layers_.reserve(n_layers);
    for (int32_t i = 0; i < n_layers; ++i) {
      layers_.emplace_back(
          register_module("layers." + std::to_string(i),
                          MiMoSdpaDecoderLayer(hidden,
                                               n_heads,
                                               ffn,
                                               /*max_seq=*/group_size,
                                               eps,
                                               rope_theta)));
    }
    norm_ = register_module("norm", MiMoSimpleRMSNorm(hidden, eps));
  }

  torch::Tensor forward(const torch::Tensor& embeds) {
    torch::Tensor h = embeds;
    for (auto& layer : layers_) {
      // No KV-cache: each group is encoded in a single full-sequence pass.
      // KV return values are discarded; only the hidden output is kept.
      torch::Tensor dummy_k, dummy_v;
      auto result = layer->forward(h, dummy_k, dummy_v, is_causal_);
      h = std::get<0>(result);
    }
    return norm_->forward(h);
  }

  void load_state_dict(const StateDict& sd) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          sd.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(sd.get_dict_with_prefix("norm."));
  }

 private:
  bool is_causal_;
  std::vector<MiMoSdpaDecoderLayer> layers_;
  MiMoSimpleRMSNorm norm_{nullptr};
};
TORCH_MODULE(MiMoSpeechGroupEncoder);

// ─── MiMoSpeechEncoderImpl ───────────────────────────────────────────────────
// Full speech-encoding pipeline:
//   RVQ codes [T, C]  →  sum speech embeddings  →  group encoder  →  project
//                                                                  →  [T/gs, H]
// Encapsulates all parameters that are audio-modality-specific, making it
// straightforward to reuse in future SLM variants.

class MiMoSpeechEncoderImpl final : public torch::nn::Module {
 public:
  MiMoSpeechEncoderImpl(int32_t audio_channels,
                        const std::vector<int64_t>& speech_vocab_sizes,
                        const std::vector<int32_t>& speech_empty_ids,
                        int32_t group_size,
                        int32_t group_encoder_layers,
                        int32_t group_encoder_heads,
                        int64_t embed_dim,
                        int64_t global_hidden,
                        bool full_attention = false,
                        float eps = 1e-6f,
                        float rope_theta = 10000.0f)
      : audio_channels_(audio_channels),
        speech_empty_ids_(speech_empty_ids),
        group_size_(group_size),
        embed_dim_(embed_dim) {
    CHECK_EQ(static_cast<int32_t>(speech_vocab_sizes.size()), audio_channels);
    CHECK_EQ(static_cast<int32_t>(speech_empty_ids.size()), audio_channels);

    speech_embeddings_.reserve(audio_channels_);
    for (int32_t c = 0; c < audio_channels_; ++c) {
      speech_embeddings_.emplace_back(register_module(
          "speech_embeddings." + std::to_string(c),
          torch::nn::Embedding(
              torch::nn::EmbeddingOptions(speech_vocab_sizes[c], embed_dim)
                  .padding_idx(speech_empty_ids[c]))));
    }

    group_encoder_ =
        register_module("input_local_transformer",
                        MiMoSpeechGroupEncoder(group_encoder_layers,
                                               embed_dim,
                                               group_encoder_heads,
                                               embed_dim * 4,
                                               group_size_,
                                               full_attention,
                                               eps,
                                               rope_theta));

    group_downcast_ = register_module(
        "speech_group_downcast",
        torch::nn::Linear(
            torch::nn::LinearOptions(group_size * embed_dim, global_hidden)
                .bias(false)));
  }

  // codes: [T, audio_channels]   T must be divisible by group_size
  // Returns: [T_groups, global_hidden]
  torch::Tensor encode(const torch::Tensor& codes) {
    const int64_t T = codes.size(0);
    const int64_t T_groups = T / group_size_;

    // reshape() tolerates non-contiguous input; view() does not.
    auto c = codes.reshape({T_groups, group_size_, audio_channels_})
                 .to(torch::kInt32);

    // Use the actual embedding weight dtype (set during load_model) rather than
    // hardcoding bfloat16, so float16 and float32 inference both work
    // correctly.
    const auto emb_dtype = speech_embeddings_[0]->weight.dtype();
    auto embeds = torch::zeros(
        {T_groups, group_size_, embed_dim_},
        torch::TensorOptions().dtype(emb_dtype).device(codes.device()));

    for (int32_t ch = 0; ch < audio_channels_; ++ch) {
      auto ids = c.select(/*dim=*/2, ch);  // [T_groups, group_size]
      auto safe_ids = ids.clamp(0, speech_embeddings_[ch]->weight.size(0) - 1);
      auto emb = speech_embeddings_[ch]->forward(
          safe_ids);  // [T_groups, group_size, D]
      auto mask = (ids == speech_empty_ids_[ch]).unsqueeze(-1);
      emb.masked_fill_(mask, 0.0f);
      embeds += emb;
    }

    // Encode each group independently: [T_groups, group_size, D]
    auto encoded = group_encoder_->forward(embeds);

    // Project group to global hidden size: [T_groups, H]
    return group_downcast_->forward(
        encoded.reshape({T_groups, group_size_ * embed_dim_}));
  }

  // Expose embedding tables so the local transformer can share them.
  const std::vector<torch::nn::Embedding>& speech_embeddings() const {
    return speech_embeddings_;
  }

  void load_state_dict(const StateDict& sd) {
    for (int32_t c = 0; c < audio_channels_; ++c) {
      auto w =
          sd.get_tensor("speech_embeddings." + std::to_string(c) + ".weight");
      if (w.defined()) {
        speech_embeddings_[c]->weight.data().copy_(w);
      }
    }
    group_encoder_->load_state_dict(
        sd.get_dict_with_prefix("input_local_transformer."));
    auto gd = sd.get_tensor("speech_group_downcast.weight");
    if (gd.defined()) {
      group_downcast_->weight.data().copy_(gd);
    }
  }

 private:
  int32_t audio_channels_;
  std::vector<int32_t> speech_empty_ids_;
  int32_t group_size_;
  int64_t embed_dim_;

  std::vector<torch::nn::Embedding> speech_embeddings_;
  MiMoSpeechGroupEncoder group_encoder_{nullptr};
  torch::nn::Linear group_downcast_{nullptr};
};
TORCH_MODULE(MiMoSpeechEncoder);

// ─── MiMoLocalTransformerImpl
// ───────────────────────────────────────────────── Optional TTS component:
// autoregressively generates speech tokens from the global hidden state
// produced after each text token step.
//
// Maps to: local_transformer in the Python model.

class MiMoLocalTransformerImpl final : public torch::nn::Module {
 public:
  MiMoLocalTransformerImpl(int32_t n_layers,
                           int64_t hidden,
                           int32_t n_heads,
                           int64_t ffn,
                           float eps = 1e-6f,
                           float rope_theta = 10000.0f) {
    static constexpr int64_t kMaxLocalSeq = 32;
    layers_.reserve(n_layers);
    for (int32_t i = 0; i < n_layers; ++i) {
      layers_.emplace_back(register_module(
          "layers." + std::to_string(i),
          MiMoSdpaDecoderLayer(
              hidden, n_heads, ffn, kMaxLocalSeq, eps, rope_theta)));
    }
    norm_ = register_module("norm", MiMoSimpleRMSNorm(hidden, eps));
  }

  // Greedy autoregressive decode of speech tokens for one text position.
  //
  // initial_embed: [B, 1, local_dim]   downcast of the global hidden state
  // group_size, delay_pattern, speech_empty_ids: model hyper-parameters
  // speech_embeddings, lm_heads: shared from MiMoAudioForCausalLM
  // embed_to_local: optional projection input_local_dim → local_dim
  //
  // Returns: [B, group_size, audio_channels]
  torch::Tensor local_forward(
      const torch::Tensor& initial_embed,
      int32_t group_size,
      const std::vector<int32_t>& delay_pattern,
      const std::vector<int32_t>& speech_empty_ids,
      std::vector<torch::nn::Embedding> speech_embeddings,
      std::vector<torch::nn::Linear> lm_heads,
      torch::nn::Linear embed_to_local) {
    const int64_t B = initial_embed.size(0);
    const int32_t audio_ch = static_cast<int32_t>(delay_pattern.size());
    const int32_t max_delay =
        *std::max_element(delay_pattern.begin(), delay_pattern.end());
    const int32_t iters = group_size + max_delay;

    auto result = torch::zeros(
        {B, static_cast<int64_t>(group_size), static_cast<int64_t>(audio_ch)},
        torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(initial_embed.device()));

    // Per-layer incremental KV-cache
    std::vector<torch::Tensor> past_ks(layers_.size());
    std::vector<torch::Tensor> past_vs(layers_.size());

    torch::Tensor embed = initial_embed;  // [B, 1, local_dim]

    for (int32_t t = 0; t < iters; ++t) {
      torch::Tensor h = embed;
      for (size_t li = 0; li < layers_.size(); ++li) {
        auto [out, nk, nv] = layers_[li]->forward(h,
                                                  past_ks[li],
                                                  past_vs[li],
                                                  /*is_causal=*/false);
        past_ks[li] = nk;
        past_vs[li] = nv;
        h = out;
      }
      auto hidden = norm_->forward(h);  // [B, 1, local_dim]

      embed = torch::zeros_like(initial_embed);

      for (int32_t c = 0; c < audio_ch; ++c) {
        if (delay_pattern[c] <= t && t < delay_pattern[c] + group_size) {
          auto logits = lm_heads[c]->forward(hidden.squeeze(1));  // [B, vocab]
          logits.select(/*dim=*/-1, speech_empty_ids[c])
              .fill_(-std::numeric_limits<float>::infinity());
          // NOTE: greedy decode only.  For high-quality TTS, temperature /
          // top-k / top-p sampling (as in MiMoSampler) should be added here.
          auto token = torch::argmax(logits, -1);  // [B]
          result.select(1, t - delay_pattern[c]).select(-1, c).copy_(token);

          auto emb =
              speech_embeddings[c]->forward(token.unsqueeze(1));  // [B, 1, D]
          embed +=
              (embed_to_local.is_empty()) ? emb : embed_to_local->forward(emb);
        }
      }
    }
    return result;  // [B, group_size, audio_channels]
  }

  void load_state_dict(const StateDict& sd) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          sd.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(sd.get_dict_with_prefix("norm."));
  }

 private:
  std::vector<MiMoSdpaDecoderLayer> layers_;
  MiMoSimpleRMSNorm norm_{nullptr};
};
TORCH_MODULE(MiMoLocalTransformer);

// ─── MiMoAudioModelImpl
// ─────────────────────────────────────────────────────── Global Qwen2
// transformer – reuses LlmModelImplBase unchanged. The base-class forward()
// already handles params.embedding.input_embedding (pre-computed by
// get_input_embeddings), so no override is needed.

class MiMoAudioModelImpl final
    : public LlmModelImplBase<layer::Qwen2DecoderLayer> {
 public:
  explicit MiMoAudioModelImpl(const ModelContext& ctx)
      : LlmModelImplBase<layer::Qwen2DecoderLayer>("mimo_audio",
                                                   ctx.get_model_args()) {
    const auto& args = ctx.get_model_args();
    norm_ = register_module("norm", layer::RMSNorm(ctx));
    embed_tokens_ = register_module("embed_tokens", layer::WordEmbedding(ctx));
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); ++i) {
      layers_.emplace_back(register_module("layers." + std::to_string(i),
                                           layer::Qwen2DecoderLayer(ctx, i)));
    }
  }
};
TORCH_MODULE(MiMoAudioModel);

// ─── MiMoAudioForCausalLMImpl
// ───────────────────────────────────────────────── Top-level model. Implements
// the three-method VLM interface so it can be registered with
// REGISTER_CAUSAL_VLM_MODEL and served through the standard xLLM VLM pipeline
// without any scheduler changes.

class MiMoAudioForCausalLMImpl final : public torch::nn::Module {
 public:
  explicit MiMoAudioForCausalLMImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    const auto& opt = ctx.get_tensor_options();

    audio_channels_ = args.audio_channels();
    group_size_ = args.group_size();
    speech_empty_ids_ = args.speech_empty_ids();
    delay_pattern_ = args.delay_pattern();
    speech_empty_token_id_ = args.speech_empty_token_id();

    const int64_t hidden = args.hidden_size();
    const int64_t local_dim = args.local_dim();
    const int64_t input_local_dim =
        args.input_local_dim() > 0 ? args.input_local_dim() : local_dim;

    // ── Global Qwen2 transformer ────────────────────────────────────────────
    model_ = register_module("model", MiMoAudioModel(ctx));

    // ── Text LM head ────────────────────────────────────────────────────────
    lm_head_ = register_module("lm_head", layer::LmHead(ctx));

    // ── Speech encoder (audio modality → global embedding) ──────────────────
    // NOTE on naming: the module is registered as "speech_encoder" in the C++
    // module tree, but its sub-module weights live at the root level of the
    // HuggingFace checkpoint (speech_embeddings.*, input_local_transformer.*,
    // speech_group_downcast.*).  load_model() bypasses the tree and maps those
    // keys directly, so inference is unaffected.  If serialisation (save/load
    // via state_dict()) is ever needed, re-register the sub-modules at the
    // top level instead of using this "speech_encoder" wrapper.
    speech_encoder_ = register_module(
        "speech_encoder",
        MiMoSpeechEncoder(audio_channels_,
                          args.speech_vocab_sizes(),
                          speech_empty_ids_,
                          group_size_,
                          args.input_local_layers(),
                          args.local_attn_heads(),
                          input_local_dim,
                          hidden,
                          args.input_full_attention(),
                          static_cast<float>(args.rms_norm_eps()),
                          args.rope_theta()));

    // ── TTS-only components (optional) ──────────────────────────────────────
    // Constructed only when local_layers > 0 (set via config "local_layers").
    // Set local_layers: 0 (the default) for ASR-only deployments to avoid
    // allocating the ~400 M-parameter local transformer on the GPU.
    const int32_t local_layers = args.local_layers();
    if (local_layers > 0) {
      hidden_states_downcast_ = register_module(
          "hidden_states_downcast",
          torch::nn::Linear(
              torch::nn::LinearOptions(hidden, local_dim).bias(false)));
      hidden_states_downcast_->weight.set_data(
          hidden_states_downcast_->weight.to(opt));

      local_transformer_ = register_module(
          "local_transformer",
          MiMoLocalTransformer(local_layers,
                               local_dim,
                               args.local_attn_heads(),
                               args.local_ffn_dim(),
                               static_cast<float>(args.rms_norm_eps()),
                               args.rope_theta()));

      local_lm_heads_.reserve(audio_channels_);
      const auto& vocab_sizes = args.speech_vocab_sizes();
      for (int32_t c = 0; c < audio_channels_; ++c) {
        const int64_t vocab = vocab_sizes.empty() ? 1025 : vocab_sizes[c];
        auto head = register_module(
            "local_transformer_lm_heads." + std::to_string(c),
            torch::nn::Linear(
                torch::nn::LinearOptions(local_dim, vocab).bias(false)));
        head->weight.set_data(head->weight.to(opt));
        local_lm_heads_.emplace_back(head);
      }

      if (input_local_dim != local_dim) {
        speech_embeddings_to_local_ =
            register_module("speech_embeddings_to_local",
                            torch::nn::Linear(torch::nn::LinearOptions(
                                                  input_local_dim, local_dim)
                                                  .bias(false)));
        speech_embeddings_to_local_->weight.set_data(
            speech_embeddings_to_local_->weight.to(opt));
      }
    }
  }

  // ── VLM interface ──────────────────────────────────────────────────────────

  // Step 2 (see file header): audio codes → speech-group embeddings.
  // Called by the serving framework before get_input_embeddings().
  MMDict get_multimodal_embeddings(const ModelInputParams& params) {
    auto codes_opt =
        params.multimodal.mm_data.get<torch::Tensor>("audio|codes");
    if (!codes_opt.has_value()) {
      return {};  // Text-only request
    }
    auto speech_embeds =
        speech_encoder_->encode(codes_opt.value());  // [T_groups, hidden]
    MMDict result;
    result["audio|embedding"] = speech_embeds;
    return result;
  }

  // Step 3 (see file header): merge text embedding + speech-group embeddings.
  // Replaces each <|empty|> position in the token sequence with the
  // corresponding speech-group embedding — the exact same merge pattern as
  // Qwen2-VL uses for visual tokens.
  torch::Tensor get_input_embeddings(const torch::Tensor& input_ids,
                                     const ModelInputParams& params) {
    // Text embeddings for all tokens: [num_tokens, hidden]
    auto text_embeds = model_->get_input_embeddings(input_ids);

    auto audio_emb_opt =
        params.multimodal.mm_data.get<torch::Tensor>("audio|embedding");
    if (!audio_emb_opt.has_value()) {
      return text_embeds;
    }
    const auto& audio_embeds = audio_emb_opt.value();  // [T_groups, hidden]

    // Find the <|empty|> positions (one per speech group) and replace them.
    auto empty_pos = (input_ids == speech_empty_token_id_)
                         .nonzero()
                         .squeeze(-1);  // [T_groups]

    CHECK_EQ(empty_pos.size(0), audio_embeds.size(0))
        << "Token count mismatch: " << empty_pos.size(0)
        << " <|empty|> tokens vs " << audio_embeds.size(0) << " audio groups";

    text_embeds.index_put_({empty_pos}, audio_embeds.to(text_embeds.dtype()));
    return text_embeds;
  }

  // Step 4: delegate to global Qwen2 transformer.
  // LlmModelImplBase::forward() uses params.embedding.input_embedding when
  // set by the serving framework, so no special logic is needed here.
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) {
    return model_(tokens, positions, kv_caches, params);
  }

  // Step 5: text LM head.
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    auto h = selected_idxes.defined()
                 ? hidden_states.index_select(0, selected_idxes)
                 : hidden_states;
    return lm_head_(h);
  }

  // ── Optional TTS step ──────────────────────────────────────────────────────
  // Not called during standard ASR serving.  Used for speech synthesis.
  // Requires local_layers > 0 in the model config; call is_tts_enabled() first.
  bool is_tts_enabled() const { return !local_transformer_.is_empty(); }

  // global_hidden: [B, 1, hidden_size]
  torch::Tensor local_forward(const torch::Tensor& global_hidden) {
    CHECK(is_tts_enabled())
        << "local_forward() called but TTS components are not built. "
           "Set local_layers > 0 in the model config to enable TTS.";
    auto initial_embed =
        hidden_states_downcast_->forward(global_hidden);  // [B, 1, local_dim]
    // Module holders are shared_ptr wrappers — copying them is cheap.
    return local_transformer_->local_forward(
        initial_embed,
        group_size_,
        delay_pattern_,
        speech_empty_ids_,
        speech_embeddings_from_encoder(),  // const ref → vector copy in callee
        local_lm_heads_,
        speech_embeddings_to_local_);
  }

  // ── Weight loading ─────────────────────────────────────────────────────────
  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& sd : loader->get_state_dicts()) {
      // Global Qwen2 (model.*)
      model_->load_state_dict(sd->get_dict_with_prefix(std::vector<std::string>{
          "model.language_model.", "language_model.model.", "model.", ""}));

      // Text LM head
      lm_head_->load_state_dict(sd->get_dict_with_prefix(
          std::vector<std::string>{"lm_head.", "model.lm_head."}));

      // Speech encoder: speech_embeddings.*, input_local_transformer.*,
      //                 speech_group_downcast.*
      speech_encoder_->load_state_dict(*sd);

      // TTS components – only when local_layers > 0
      if (is_tts_enabled()) {
        load_weight(*sd,
                    "hidden_states_downcast.weight",
                    hidden_states_downcast_->weight);
        local_transformer_->load_state_dict(
            sd->get_dict_with_prefix("local_transformer."));
        for (int32_t c = 0; c < audio_channels_; ++c) {
          load_weight(
              *sd,
              "local_transformer_lm_heads." + std::to_string(c) + ".weight",
              local_lm_heads_[c]->weight);
        }
        if (!speech_embeddings_to_local_.is_empty()) {
          load_weight(*sd,
                      "speech_embeddings_to_local.weight",
                      speech_embeddings_to_local_->weight);
        }
      }
    }
  }

  // Accessors used by the TTS path and by the Python inference script
  layer::LmHead& lm_head() { return lm_head_; }
  MiMoSpeechEncoder& speech_encoder() { return speech_encoder_; }
  MiMoLocalTransformer& local_transformer() { return local_transformer_; }
  torch::nn::Linear& hidden_states_downcast() {
    return hidden_states_downcast_;
  }
  int32_t audio_channels() const { return audio_channels_; }
  int32_t group_size() const { return group_size_; }
  int32_t speech_empty_token_id() const { return speech_empty_token_id_; }

  // The <|empty|> token ID comes from the tokenizer, not config.json.
  // The serving layer must call this after construction to inject the live
  // tokenizer value (tokenizer.convert_tokens_to_ids("<|empty|>")).
  void set_speech_empty_token_id(int32_t id) {
    CHECK_GE(id, 0) << "speech_empty_token_id must be a valid token ID (>= 0)";
    speech_empty_token_id_ = id;
  }

 private:
  // The speech encoder owns the embedding tables.  local_forward() copies
  // the shared_ptr handles so the tables are accessible without additional
  // memory allocation.
  const std::vector<torch::nn::Embedding>& speech_embeddings_from_encoder()
      const {
    return speech_encoder_->speech_embeddings();
  }

  static void load_weight(const StateDict& sd,
                          const std::string& key,
                          torch::Tensor& param) {
    auto w = sd.get_tensor(key);
    if (w.defined()) {
      param.data().copy_(w);
    }
  }

  int32_t audio_channels_{8};
  int32_t group_size_{4};
  int32_t speech_empty_token_id_{-1};
  std::vector<int32_t> speech_empty_ids_;
  std::vector<int32_t> delay_pattern_;

  MiMoAudioModel model_{nullptr};
  layer::LmHead lm_head_{nullptr};
  MiMoSpeechEncoder speech_encoder_{nullptr};

  // TTS-only
  torch::nn::Linear hidden_states_downcast_{nullptr};
  MiMoLocalTransformer local_transformer_{nullptr};
  std::vector<torch::nn::Linear> local_lm_heads_;
  torch::nn::Linear speech_embeddings_to_local_{nullptr};
};
TORCH_MODULE(MiMoAudioForCausalLM);

// ─── Registration
// ─────────────────────────────────────────────────────────────

REGISTER_CAUSAL_VLM_MODEL(mimo_audio, MiMoAudioForCausalLM);

REGISTER_MODEL_ARGS(mimo_audio, [&] {
  // ── Global Qwen2 transformer fields ──────────────────────────────────────
  LOAD_ARG_OR(model_type, "model_type", "mimo_audio");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6f);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR_FUNC(sliding_window, "sliding_window", [&] {
    return args->max_position_embeddings();
  });
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  // ── MiMo-V2.5-ASR specific fields ────────────────────────────────────────
  LOAD_ARG_OR(audio_channels, "audio_channels", 8);
  LOAD_ARG_OR(group_size, "group_size", 4);
  LOAD_ARG_OR(local_dim, "local_dim", 1024);
  // local_layers == 0 → TTS components are not constructed (ASR-only mode).
  // Set to 16 (or read from config) to enable the local speech decoder.
  LOAD_ARG_OR(local_layers, "local_layers", 0);
  LOAD_ARG_OR(local_attn_heads, "local_attn_heads", 64);
  LOAD_ARG_OR(local_ffn_dim, "local_ffn_dim", 4096);
  LOAD_ARG_OR(local_attn_dropout, "local_attn_dropout", 0.0f);
  LOAD_ARG_OR(input_local_layers, "input_local_layers", 6);
  // speech_empty_token_id is NOT in config.json — it comes from the tokenizer.
  // Load as a fallback; the serving layer should call
  // set_speech_empty_token_id() with the live tokenizer value after model
  // construction.
  LOAD_ARG_OR(speech_empty_token_id, "speech_empty_token_id", -1);
  LOAD_ARG_OR_FUNC(
      input_local_dim, "input_local_dim", [&] { return args->local_dim(); });
  LOAD_ARG_OR(input_full_attention, "input_full_attention", false);

  // Parse dash-separated strings  e.g. "1025-1025-129-129-129-129-129-129"
  [&]() {
    const int32_t ch = args->audio_channels();

    auto parse_i64 = [](const std::string& s,
                        int32_t n,
                        int64_t def) -> std::vector<int64_t> {
      std::vector<int64_t> v;
      v.reserve(n);
      std::istringstream ss(s);
      for (std::string tok;
           std::getline(ss, tok, '-') && static_cast<int32_t>(v.size()) < n;)
        v.emplace_back(std::stoll(tok));
      while (static_cast<int32_t>(v.size()) < n) v.emplace_back(def);
      return v;
    };
    auto parse_i32 = [](const std::string& s,
                        int32_t n,
                        int32_t def) -> std::vector<int32_t> {
      std::vector<int32_t> v;
      v.reserve(n);
      std::istringstream ss(s);
      for (std::string tok;
           std::getline(ss, tok, '-') && static_cast<int32_t>(v.size()) < n;)
        v.emplace_back(static_cast<int32_t>(std::stol(tok)));
      while (static_cast<int32_t>(v.size()) < n) v.emplace_back(def);
      return v;
    };

    auto s64 = json.value<std::string>("speech_vocab_size");
    args->speech_vocab_sizes() =
        s64 ? parse_i64(*s64, ch, 1025LL) : std::vector<int64_t>(ch, 1025LL);

    auto s32 = json.value<std::string>("speech_zeroemb_idx");
    args->speech_empty_ids() =
        s32 ? parse_i32(*s32, ch, 1024) : std::vector<int32_t>(ch, 1024);

    auto dp = json.value<std::string>("delay_pattern");
    if (dp) {
      args->delay_pattern() = parse_i32(*dp, ch, 0);
    } else {
      std::vector<int32_t> v(ch);
      std::iota(v.begin(), v.end(), 0);
      args->delay_pattern() = v;
    }
  }();
});

// ─── Multimodal processor
// ─────────────────────────────────────────────────────
#include "processors/mimo_audio_processor.h"
#include "processors/mimo_audio_prompt_processor.h"

using MiMoAudioMultimodalProcessor =
    MultimodalProcessor<MiMoAudioPromptProcessor,
                        ImageNoneProcessor,
                        VideoNoneProcessor,
                        MiMoAudioProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(mimo_audio, MiMoAudioMultimodalProcessor);

}  // namespace xllm
