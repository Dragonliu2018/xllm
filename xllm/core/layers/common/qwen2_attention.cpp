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

#include "qwen2_attention.h"

#include <glog/logging.h>

#include <tuple>

namespace {
inline bool is_qwen3_model(const std::string& model_type) {
  static const std::set<std::string> qwen3_type_set = {
      "qwen3", "qwen3_vl", "qwen3_moe", "qwen3_vl_moe"};
  return qwen3_type_set.contains(model_type);
}
}  // namespace

namespace xllm {
namespace layer {

Qwen2AttentionImpl::Qwen2AttentionImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  is_qwen3_style_ = is_qwen3_model(args.model_type());
  bool qkv_bias = is_qwen3_style_ ? args.attention_bias() : true;

  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = std::sqrt(1.0f / head_dim_);

  // 1. QKV parallel linear
  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                num_heads_,
                                                num_kv_heads_,
                                                args.head_dim(),
                                                num_kv_head_replicas_,
                                                qkv_bias,
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options));

  // 2. Output projection
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * args.head_dim(),
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  // 3. RMSNorm
  if (is_qwen3_style_) {
    q_norm_ = register_module(
        "q_norm", RMSNorm(args.head_dim(), args.rms_norm_eps(), options));

    k_norm_ = register_module(
        "k_norm", RMSNorm(args.head_dim(), args.rms_norm_eps(), options));
  }

  // 4. Rotary embedding
  rotary_emb_ =
      register_module("rope",
                      MRotaryEmbedding(/*rotary_dim=*/head_dim_,
                                       args.max_position_embeddings(),
                                       args.rope_theta(),
                                       /*interleaved=*/false,
                                       args.rope_scaling_mrope_section(),
                                       options));

  // 5. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor Qwen2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // DEBUG: Check if this method is being called
  LOG(WARNING) << "[QWEN2_ATTENTION_DEBUG] Qwen2AttentionImpl::forward called, "
                  "attn_mask.defined(): "
               << attn_metadata.attn_mask.defined();

  // 1. qkv projection
  // Ensure input dtype matches weight dtype before calling forward
  torch::Tensor hidden_states_converted = hidden_states;
  auto qkv_weight_dtype = qkv_proj_->weight().scalar_type();
  if (hidden_states.scalar_type() != qkv_weight_dtype) {
    hidden_states_converted = hidden_states.to(qkv_weight_dtype);
  }
  auto qkv = qkv_proj_->forward(hidden_states_converted);
  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  const int64_t T = q.size(0);

  if (is_qwen3_style_) {
    // 2. q-norm
    // Ensure input dtype matches weight dtype before calling forward
    auto q_weight_dtype = q_norm_->weight().scalar_type();
    auto q_original_dtype = q.scalar_type();
    torch::Tensor q_for_norm = q;
    if (q.scalar_type() != q_weight_dtype) {
      q_for_norm = q.to(q_weight_dtype);
    }
    q = std::get<0>(q_norm_->forward(q_for_norm));
    // Convert back to original dtype if needed
    if (q.scalar_type() != q_original_dtype) {
      q = q.to(q_original_dtype);
    }

    // 3. k-norm
    auto k_weight_dtype = k_norm_->weight().scalar_type();
    auto k_original_dtype = k.scalar_type();
    torch::Tensor k_for_norm = k;
    if (k.scalar_type() != k_weight_dtype) {
      k_for_norm = k.to(k_weight_dtype);
    }
    k = std::get<0>(k_norm_->forward(k_for_norm));
    // Convert back to original dtype if needed
    if (k.scalar_type() != k_original_dtype) {
      k = k.to(k_original_dtype);
    }
  }

  // 4. rope
  rotary_emb_->forward(q, k, positions, attn_metadata);
  q = q.view({T, q_size_});
  k = k.view({T, kv_size_});

  // Debug: Log Q, K, V for token 0 and 36 - to compare with diffusers
  // Note: v doesn't get RoPE, so it's unchanged after qkv_proj
  if (T > 0) {
    auto q0 = q[0].slice(0, 0, std::min(10L, q.size(-1)));
    auto k0 = k[0].slice(0, 0, std::min(10L, k.size(-1)));
    auto v0 = v[0].slice(0, 0, std::min(10L, v.size(-1)));
    LOG(INFO) << "[LongCatImage] [DEBUG] Q (query, after RoPE) token 0 first "
                 "10 dims: "
              << q0;
    LOG(INFO)
        << "[LongCatImage] [DEBUG] K (key, after RoPE) token 0 first 10 dims: "
        << k0;
    LOG(INFO)
        << "[LongCatImage] [DEBUG] V (value, no RoPE) token 0 first 10 dims: "
        << v0;
  }
  if (T > 36) {
    auto q36 = q[36].slice(0, 0, std::min(10L, q.size(-1)));
    auto k36 = k[36].slice(0, 0, std::min(10L, k.size(-1)));
    auto v36 = v[36].slice(0, 0, std::min(10L, v.size(-1)));
    LOG(INFO) << "[LongCatImage] [DEBUG] Q (query, after RoPE) token 36 first "
                 "10 dims: "
              << q36;
    LOG(INFO)
        << "[LongCatImage] [DEBUG] K (key, after RoPE) token 36 first 10 dims: "
        << k36;
    LOG(INFO)
        << "[LongCatImage] [DEBUG] V (value, no RoPE) token 36 first 10 dims: "
        << v36;
  }

  // 5. store k/v cache and do attention
  LOG(WARNING) << "[QWEN2_ATTENTION_DEBUG] About to call attn_->forward with "
                  "attn_mask.defined(): "
               << attn_metadata.attn_mask.defined()
               << ", max_seq_len: " << attn_metadata.max_seq_len;
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  LOG(WARNING) << "[QWEN2_ATTENTION_DEBUG] Returned from attn_->forward";

  // Debug: Log attention output for layer 0, token 0 and 36
  if (out.size(0) > 36) {
    int64_t hd = out.size(-1);
    auto attn_out_0 = out[0].slice(0, 0, std::min(10L, hd));
    auto attn_out_36 = out[36].slice(0, 0, std::min(10L, hd));
    LOG(INFO) << "[LongCatImage] [DEBUG] Attention output (before o_proj) - "
                 "token 0 first 10 dims: "
              << attn_out_0;
    LOG(INFO) << "[LongCatImage] [DEBUG] Attention output (before o_proj) - "
                 "token 36 first 10 dims: "
              << attn_out_36;
  }

  // 6. output projection. Match diffusers: keep bf16 (no float32).
  return o_proj_->forward(out);
}

void Qwen2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (is_qwen3_style_) {
    q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
    k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
  }
}

}  // namespace layer
}  // namespace xllm
