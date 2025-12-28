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

#include <limits>

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
      // Debug: Print mask info
      LOG(INFO) << "[batch_prefill] Received attention_mask with shape: "
                << m.sizes() << ", dtype: " << m.dtype();

      // Ensure mask is on the same device as query
      auto device = query.device();
      if (m.device() != device) {
        m = m.to(device);
      }
      // Ensure mask is float type for FlashInfer
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

      // Debug: Verify causal mask is correct
      LOG(INFO) << "[batch_prefill] Causal mask verification: "
                << "causal[0,0]=" << causal_mask[0][0].item<float>()
                << ", causal[0,1]=" << causal_mask[0][1].item<float>()
                << ", causal[1,0]=" << causal_mask[1][0].item<float>()
                << ", causal[1,1]=" << causal_mask[1][1].item<float>();

      // Expand padding mask to 2D: padding_mask_2d[i, j] = m[j] for all i
      // This broadcasts the 1D key mask to all query rows
      auto padding_mask_2d = m.unsqueeze(0).expand({seq_len, seq_len});

      // Combine masks: position j is attendable from position i if
      // j <= i (causal) AND j is not padding (padding_mask[j] = 1)
      auto combined_mask = causal_mask * padding_mask_2d;

      // Convert from (1=attend, 0=mask_out) to additive bias format
      // (0=attend, large_negative=mask_out) for proper softmax zero-out
      // Note: Using -65504.0f (max representable in float16) instead of -inf
      // to avoid potential NaN issues when FlashInfer converts to bfloat16
      // This matches HuggingFace's approach of using dtype.min for masking
      const float mask_value = -65504.0f;  // Safe for both float16 and bfloat16
      auto converted_mask =
          torch::where(combined_mask > 0.5f,
                       torch::zeros_like(combined_mask),
                       torch::full_like(combined_mask, mask_value));

      // Flatten the 2D mask to 1D for FlashInfer's packed format
      // IMPORTANT: FlashInfer expects mask in row-major order:
      // flat_mask[i * kv_len + j] = whether query i can attend to key j
      // Since our converted_mask is already [query, key] format, we can flatten
      // directly Use contiguous() to ensure proper memory layout for FlashInfer
      auto flat_mask = converted_mask.contiguous().view({-1}).contiguous();

      // Convert mask to match query dtype for consistency with FlashInfer
      // FlashInfer may expect mask in same dtype as query/key/value
      if (flat_mask.dtype() != query.dtype() && query.is_floating_point()) {
        flat_mask = flat_mask.to(query.dtype());
        LOG(INFO) << "[batch_prefill] Converted mask dtype to match query: "
                  << query.dtype();
      }
      processed_mask = flat_mask;

      // Create mask_indptr for single sequence batching
      // For a single sequence of length N, the 2D mask has N*N elements
      // mask_indptr = [0, N*N] indicates the start and end of the mask
      auto mask_indptr = torch::zeros(
          {2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      mask_indptr[0] = 0;
      mask_indptr[1] = static_cast<int32_t>(seq_len * seq_len);
      mask_indptr_opt = mask_indptr;

      // Debug: Print processed mask statistics
      LOG(INFO) << "[batch_prefill] Expanded mask from 1D [" << seq_len
                << "] to 2D [" << seq_len << ", " << seq_len
                << "], flattened to [" << flat_mask.size(0) << "]";
      // Use float32 copy for min/max to avoid dtype issues
      auto mask_float = flat_mask.to(torch::kFloat32);
      LOG(INFO)
          << "[batch_prefill] Processed mask (converted to additive bias) min: "
          << mask_float.min().item<float>()
          << ", max: " << mask_float.max().item<float>()
          << ", dtype: " << flat_mask.dtype();
      LOG(INFO) << "[batch_prefill] mask_indptr: [0, " << seq_len * seq_len
                << "]";

      // Debug: Count masked vs unmasked positions
      auto num_attend = (mask_float > -1.0f).sum().item<int64_t>();
      auto num_masked = (mask_float < -1.0f).sum().item<int64_t>();
      LOG(INFO) << "[batch_prefill] Mask stats: attend=" << num_attend
                << ", masked=" << num_masked
                << ", total=" << (num_attend + num_masked);

      // Debug: Check first row (query 0) and last row (query seq_len-1) of the
      // 2D mask Row 0: should attend only to position 0 (causal + padding) Row
      // seq_len-1: should attend to all non-padding positions from 0 to
      // seq_len-1
      auto mask_2d = mask_float.view({seq_len, seq_len});
      auto row0_attend = (mask_2d[0] > -1.0f).sum().item<int64_t>();
      auto row_last_attend =
          (mask_2d[seq_len - 1] > -1.0f).sum().item<int64_t>();
      auto input_mask_valid = (m > 0.5f).sum().item<int64_t>();
      LOG(INFO) << "[batch_prefill] Mask row analysis: row0_attend="
                << row0_attend
                << " (expect 1 if pos 0 is valid), row_last_attend="
                << row_last_attend << " (expect " << input_mask_valid
                << " = input valid tokens)";
      LOG(INFO) << "[batch_prefill] Sample mask values: "
                << "mask_2d[0,0]=" << mask_2d[0][0].item<float>()
                << ", mask_2d[0,1]=" << mask_2d[0][1].item<float>()
                << ", mask_2d[1,0]=" << mask_2d[1][0].item<float>()
                << ", mask_2d[1,1]=" << mask_2d[1][1].item<float>();
    }
  }

  // Check if attention_mask is provided
  bool use_custom_mask = processed_mask.has_value();
  if (use_custom_mask) {
    LOG(INFO) << "[batch_prefill] Using CUSTOM mask mode (0) for attention";
  }

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
    // FA3 backend does not support custom masks yet
    // Note: mask will not be applied for FA3, only FA2 supports it
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
        /*window_left=*/window_left,
        /*mask_mode_code=*/1,  // CAUSAL - FA3 doesn't support custom masks
        /*kv_layout_code=*/0,  // NHD layout
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
