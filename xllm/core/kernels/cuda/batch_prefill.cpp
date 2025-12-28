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

#include "cuda_ops_api.h"
#include "function_factory.h"

namespace xllm::kernel::cuda {

void batch_prefill(const std::string& uri,
                   torch::Tensor plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph,
                   const std::optional<torch::Tensor>& mask) {
  // Process and validate attention mask if provided
  std::optional<torch::Tensor> processed_mask = std::nullopt;
  std::optional<torch::Tensor> mask_indptr_opt = std::nullopt;
  if (mask.has_value()) {
    auto m = mask.value();
    if (m.defined() && m.numel() > 0) {
      // Ensure mask is on the same device as query
      auto device = query.device();
      if (m.device() != device) {
        m = m.to(device);
      }
      // Ensure mask is float for constructing 2D combined mask (causal *
      // padding)
      if (!m.is_floating_point()) {
        m = m.to(torch::kFloat32);
      }

      // Get sequence length from the 1D mask
      int64_t seq_len = m.size(0);

      // FlashInfer's custom mask mode requires a 2D mask [seq_len, seq_len]
      // where mask[i, j] indicates whether query i can attend to key j.
      //
      // IMPORTANT: HuggingFace's Qwen2_5_VL uses BOTH causal masking AND
      // padding masking. The combined mask is: query i can attend to key j if:
      //   1. j <= i (causal constraint: can only attend to past/current
      //   positions)
      //   2. key j is not padding (padding_mask[j] = 1)
      //
      // Input mask format: 1 = attend (real token), 0 = mask out (padding)
      // We need to create a 2D mask that combines:
      //   - Causal mask (lower triangular): causal_mask[i, j] = 1 if j <= i
      //   - Padding mask: padding_mask[j] for each key position
      //   - Combined: combined_mask[i, j] = causal_mask[i, j] AND
      //   padding_mask[j]

      // Create causal mask (lower triangular)
      // causal_mask[i, j] = 1 if j <= i, else 0
      auto causal_mask = torch::tril(torch::ones(
          {seq_len, seq_len},
          torch::TensorOptions().dtype(torch::kFloat32).device(device)));

      // Expand padding mask to 2D: padding_mask_2d[i, j] = m[j] for all i
      // This broadcasts the 1D key mask to all query rows
      auto padding_mask_2d = m.unsqueeze(0).expand({seq_len, seq_len});

      // Combine masks: position j is attendable from position i if
      // j <= i (causal) AND j is not padding (padding_mask[j] = 1)
      auto combined_mask = causal_mask * padding_mask_2d;

      // FlashInfer custom mask is a PACKED BITMAP (see
      // include/flashinfer/attention/variants.cuh):
      // - 1 bit per (qo_idx, kv_idx), row-major offset = qo_idx * kv_len +
      // kv_idx
      // - Bit 1 = can attend, bit 0 = mask out. Packed 8 bits per byte
      // (little-endian).
      // - mask_indptr gives BYTE offsets into the packed buffer, not element
      // counts.
      const int64_t n = seq_len * seq_len;
      const int64_t num_bytes = (n + 7) / 8;
      auto flat = combined_mask.contiguous().view({-1});
      if (flat.device().type() != torch::kCPU) {
        flat = flat.cpu();
      }
      auto packed = torch::zeros(
          {num_bytes},
          torch::TensorOptions().dtype(torch::kUInt8).device(flat.device()));
      auto flat_acc = flat.accessor<float, 1>();
      auto packed_acc = packed.accessor<uint8_t, 1>();
      for (int64_t i = 0; i < n; ++i) {
        if (flat_acc[i] > 0.5f) {
          packed_acc[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
      }

      if (packed.device() != device) {
        packed = packed.to(device);
      }
      processed_mask = packed.contiguous();

      // mask_indptr: byte offsets for each batch. Single batch -> [0,
      // num_bytes].
      auto mask_indptr = torch::zeros(
          {2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      mask_indptr[0] = 0;
      mask_indptr[1] = static_cast<int32_t>(num_bytes);
      mask_indptr_opt = mask_indptr;
    }
  }

  // Check if attention_mask is provided
  bool use_custom_mask = processed_mask.has_value();

  std::string backend =
      determine_attention_backend(/*pos_encoding_mode=*/0,
                                  /*use_fp16_qk_reduction=*/false,
                                  /*use_custom_mask=*/use_custom_mask);

  if (backend == "fa2") {
    // When custom mask is provided, use CUSTOM mask mode (0) instead of CAUSAL
    // (1) Otherwise use CAUSAL for standard attention masking
    int64_t mask_mode = use_custom_mask ? 0 : 1;

    FunctionFactory::get_instance().fa2_prefill_ragged_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        key,
        value,
        q_cu_seq_lens,
        kv_cu_seq_lens,
        output,
        output_lse,
        /*mask_mode_code=*/mask_mode,  // CUSTOM (0) if mask provided, CAUSAL
                                       // (1) otherwise
        /*kv_layout_code=*/0,          // NHD layout
        window_left,
        support_pdl(),
        /*maybe_custom_mask=*/processed_mask,
        /*maybe_mask_indptr=*/mask_indptr_opt,  // Indptr for the packed 2D mask
        /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
        /*maybe_prefix_len_ptr=*/std::optional<torch::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/std::optional<torch::Tensor>(),
        /*maybe_max_item_len_ptr=*/std::optional<torch::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0,
        /*token_pos_in_items_len=*/0);
  } else if (backend == "fa3") {
    FunctionFactory::get_instance().fa3_prefill_ragged_run_func(uri).call(
        float_workspace_buffer,
        int_workspace_buffer,
        plan_info,
        query,
        key,
        value,
        q_cu_seq_lens,
        kv_cu_seq_lens,
        output,
        output_lse,
        /*mask_mode_code=*/1,  // CAUSAL
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
        support_pdl(),
        /*maybe_prefix_len_ptr=*/std::optional<torch::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/std::optional<torch::Tensor>(),
        /*maybe_max_item_len_ptr=*/std::optional<torch::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*token_pos_in_items_len=*/0);
  }
}

}  // namespace xllm::kernel::cuda
