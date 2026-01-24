# xllm 中 LongCat-Image 实现原理与推理流程

## 一、实现原理

### 1.1 总体架构

LongCat-Image 是 xllm 实现的 **DiT (Diffusion Transformer)** 类文生图管线，参考 [diffusers pipeline_longcat_image](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/longcat_image/pipeline_longcat_image.py)，对齐其行为与数值。

**核心模块：**

| 组件 | 作用 |
|------|------|
| **Text Encoder** | Qwen2.5-VL（VLM）作为条件编码器，输出 prompt embeddings |
| **Transformer** | LongCatImageTransformer2DModel，在 latent 空间做去噪 |
| **Scheduler** | FlowMatchEulerDiscreteScheduler，Flow Matching + Euler 离散步进 |
| **VAE** | 将 latent 解码为像素图像 |
| **Position Embedding** | LongCatImagePosEmbed，多轴 RoPE（text + image 2D） |

整体是 **Flow Matching** 形式的 latent diffusion：在 VAE latent 上迭代去噪，再用 VAE 解码成图。

### 1.2 设计要点

#### 1.2.1 文本编码：Qwen2.5-VL + 固定 Prompt 模板

LongCat-Image **不用** CLIP+T5，而是用 **Qwen2.5-VL** 做 text encoder，且输入必须套一层固定模板（对齐 diffusers）：

```
<|im_start|>system
As an image captioning expert, generate a descriptive text prompt ...
<|im_end|>
<|im_start|>user
{USER_PROMPT}
<|im_end|>
<|im_start|>assistant
```

流程简述：

1. 对用户 prompt 做 `split_quotation`（区分引号内/外），再 tokenize。
2. 与 **prefix**、**suffix** 拼成完整序列，pad 到 `max_sequence_length`（默认 512）。
3. 构造 `attention_mask`（1=真实 token，0=pad），用于 encoder 内 attention。
4. 调用 **Qwen2.5-VL** 的 `forward_longcat`：
   - 使用 **MROPE**：`positions` 为 `[3, seq_len]`（T/H/W 三轴，纯文本时三轴相同）。
   - 将 `attention_mask` 传入 `ModelInputParams.graph_buffer.attn_mask`，在 attention 里做加性 mask，**不**对 hidden states 乘 mask。
5. 取最后一层 hidden states，**去掉 prefix 和 suffix**，得到 `[batch, 512, hidden_size]` 的 prompt embeddings。

`forward_longcat` 与常规 forward 的差异：支持 `attention_mask`、MROPE、以及 prefill 式一次性编码，便于与 diffusers 对齐。

#### 1.2.2 位置编码：多轴 RoPE + 文本/图像分离

- **Position ID 格式**：`[modality_id, pos_h, pos_w]`
  - `modality_id`：0=文本，1=图像。
  - 文本：`pos_h = pos_w = 0..511`（序列索引）。
  - 图像：从 **512** 起算，避免与文本重叠，即 `pos_h, pos_w ∈ [512, 512+height), [512, 512+width)`（这里 height/width 是 patch 网格尺寸）。

- **LongCatImagePosEmbed**：
  - 对 `ids` 的每个轴（3 维）分别做 1D RoPE：`get_1d_rotary_pos_embed(axes_dim[i], pos_slice, theta, ...)`。
  - `axes_dims_rope` 来自配置（如 `[16, 56, 56]`），对应三轴 head 维度。
  - 输出 `(cos, sin)`，在 Transformer 的 **FluxSingleAttention** 里通过 `apply_rotary_emb` 加到 Q、K 上。

- **forward_cache**：根据 `(txt_ids, img_ids, height, width)` 缓存 RoPE 的 cos/sin；仅当 `height/width/seq_len` 变化时重算，避免重复计算。

#### 1.2.3 Latent 的 Pack / Unpack

LongCat-Image 的 latent 在进入 Transformer 前要 **pack**，解码前再 **unpack**，与 diffusers 一致。

- **pack_latents**（`[B, C, H, W]` → `[B, (H/2)(W/2), C*4]`）：
  - 按 2×2 小块折叠空间维，再拼到 channel 维。
  - `view` → `permute` → `reshape`，得到 `[B, num_patches, C*4]`。

- **unpack_latents**：逆过程，`[B, num_patches, C*4]` → `[B, C, H, W]`，再送入 VAE decoder。

VAE 的 `scale_factor`、`shift_factor`、`scaling_factor` 在 unscaling 时使用（见后文）。

#### 1.2.4 FlowMatchEulerDiscreteScheduler

- **时间**：`num_train_timesteps`（如 1000），`timesteps` 与 `sigmas` 一一对应。
- **Dynamic shifting**：`use_dynamic_shifting=true` 时，用 `mu = calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift)` 对 `sigmas` 做 `time_shift`，使调度随 **图像序列长度**（即分辨率）变化。
- **step**：根据当前 `timestep` 取 `sigma`，用 Euler 更新 `latent`；支持 `per_token_timesteps`（LongCat-Image 未用）。

#### 1.2.5 分类器自由引导（CFG）

- 当 `guidance_scale > 1` 时启用 CFG：
  - 用 **正 prompt** 和 **负 prompt**（空串或用户指定）分别 encode 得到 `encoded_prompt_embeds`、`negative_encoded_embeds`。
  - 对 **同一批 latent**，分别用正/负条件各做一次 Transformer forward（负向用 `step_idx = i + 10000` 避免与正向共享缓存）。
  - 更新：`noise_pred = neg + guidance_scale * (pos - neg)`。
- **CFG Renorm**（`enable_cfg_renorm`）：
  - 用条件分支的范数对 CFG 后的 `noise_pred` 做缩放，避免过大，数值上更稳定。

### 1.3 模型注册与加载

- **DiT 注册**：`ModelRegistry::register_dit_model_factory("LongCat-Image", ...)`，直接以 `"LongCat-Image"` 含连字符的名字注册。
- **加载**：`DiTModelLoader` 按 `model_index` 等约定，拆出 `transformer`、`vae`、`text_encoder`、`tokenizer`、`text_processor`、`scheduler` 等子模块，各自 `load_model` / `load_state_dict`。
- **Text encoder**：单卡且 `tp_group == nullptr` 时，会为 VLM 创建独立的 `ProcessGroup`，再构造 `Qwen2_5_VLForConditionalGeneration` 并加载权重。

---

## 二、推理流程

### 2.1 端到端调用链

```
DiT API Request
    ↓
DiTEngine::step(batches)
    ↓
DiTWorker::prepare_inputs(batch) → DiTForwardInput
    ↓
DiTWorker::step(input) → DiTExecutor::forward(input)
    ↓
LongCatImagePipeline::forward(DiTForwardInput)
    ↓
forward_(...)  // 内部实现
    ↓
[encode_prompt] → [prepare_latents] → [timesteps] → [pos_embed forward_cache]
    ↓
Denoising Loop (N steps)
    ↓
[unpack_latents] → [VAE decode] → [postprocess] → DiTForwardOutput
```

### 2.2 详细步骤

#### Step 1：解析输入与 CFG

- 从 `DiTForwardInput` 取出 `prompts`、`negative_prompts`、`generation_params`（如 `height`、`width`、`num_inference_steps`、`guidance_scale`、`seed` 等）。
- 若 `guidance_scale > 1`，则 `do_classifier_free_guidance = true`；若无 `negative_prompts`，则用空串。

#### Step 2：Encode Prompt

- **encode_prompt**：
  - 若未提供 `prompt_embeds`，则 **encode_prompt_qwen**：
    - Tokenize + 模板拼接 + padding，得到 `input_ids`、`attention_mask`。
    - MROPE 的 `positions`：`attention_mask.cumsum(-1) - 1`，pad 位置填 1，再 `expand` 成 `[3, seq_len]`。
    - `forward_longcat(tokens, positions, kv_caches, input_params, attention_mask)` → 取最后一层，去掉 prefix/suffix → `[B, 512, hidden_size]`。
  - **prepare_text_ids**：为 512 个文本 token 生成 `[num_tokens, 3]` 的 position IDs（modality=0, pos_h=pos_w=0..511），float32。
- 若 CFG，对负 prompt 同样走一遍，得到 `negative_encoded_embeds`、`negative_text_ids`。

#### Step 3：Prepare Latents

- **prepare_latents**：
  - 按 `height`、`width`、`vae_scale_factor` 得到 `adjusted_height`、`adjusted_width`。
  - 若无传入 `latents`：`randn_tensor(shape, seed)` → **pack_latents** → `[B, num_patches, C*4]`。
  - **prepare_latent_image_ids**：图像 patch 的 position IDs，`modality=1`，`pos_h`/`pos_w` 从 512 起，整型先算再转 float32，避免 bf16 精度问题。

#### Step 4：Timesteps 与 Scheduler

- 自定义 `sigmas`：`1 - i/(steps-1)*(1 - 1/steps)`，共 `num_inference_steps` 个。
- `image_seq_len = prepared_latents.size(1)`，算 `mu = calculate_shift(...)`。
- **retrieve_timesteps**：`scheduler->set_timesteps(..., sigmas, mu)`，得到 `timesteps`。
- `scheduler->set_begin_index(0)`。

#### Step 5：Position Embedding 缓存

- **pos_embed_->forward_cache(text_ids, latent_image_ids, height, width)**：
  - 拼接 `txt_ids` 与 `img_ids`，调用 `forward` 得到多轴 RoPE 的 cos/sin。
  - 若 `height`/`width`/`seq_len` 与上次相同则用缓存。
- `image_rotary_emb = stack(rot_emb1, rot_emb2)`，形状 `[2, text_seq + image_seq, head_dim]` 等，供 Transformer 使用。

#### Step 6：Denoising Loop

对每个 `i`：

1. **timestep**：`t = timesteps[i]`，缩放到 `[0,1]` 区间（如 `/1000`）得到 `timestep`。
2. **Transformer forward**：
   - `noise_pred = transformer->forward(prepared_latents, encoded_prompt_embeds, timestep, image_rotary_emb)`。
   - Transformer 内部：`x_embedder`(latents) + `context_embedder`(prompt_embeds) + `time_embed`(timestep)；再过 `transformer_blocks` 与 `single_transformer_blocks`（每块里 Attn 使用 `image_rotary_emb`）；最后 `norm_out` + `proj_out` → 预测噪声。
3. **CFG**（若开启）：
   - 用 `negative_encoded_embeds` 再做一次 `transformer->forward(..., step_idx = i+10000)` → `negative_noise_pred`。
   - `noise_pred = neg + guidance_scale * (pos - neg)`。
   - 若 `enable_cfg_renorm`，再按条件分支范数做 renorm。
4. **Scheduler step**：`prev_latents = scheduler->step(noise_pred, t, prepared_latents)`，更新 `prepared_latents`。

循环结束后，`prepared_latents` 即为去噪后的 packed latent。

#### Step 7：VAE Decode 与后处理

- **unpack_latents**：将 `[B, num_patches, C*4]` 还原为 `[B, C, H, W]`。
- **VAE 逆缩放**：`latents = latents / vae_scaling_factor + vae_shift_factor`。
- **VAE decode**：`image = vae->decode(latents)`。
- **VAEImageProcessor::postprocess**：例如 denormalize 到 [0,1]、clip、reshape 等。
- 最终转为 `cpu()`、`float32`，按 batch 切分成 `DiTForwardOutput.tensors`。

### 2.3 数据形状摘要

| 阶段 | 张量 | 典型形状 |
|------|------|----------|
| Prompt 编码后 | `prompt_embeds` | `[B, 512, joint_attention_dim]` |
| 文本 position IDs | `text_ids` | `[512, 3]` |
| 初始 latent（packed） | `prepared_latents` | `[B, (H/2)(W/2), 64]`（C=16 → 64） |
| 图像 position IDs | `latent_image_ids` | `[num_patches, 3]` |
| RoPE 缓存 | `image_rotary_emb` | `[2, text_len+img_len, head_dim]` 等 |
| Transformer 噪声预测 | `noise_pred` | 与 `prepared_latents` 同形 |
| VAE 输入 | `unpacked_latents` | `[B, 16, H', W']` |
| 最终图像 | `image` | `[B, 3, height, width]` |

### 2.4 与 diffusers 的对应关系

- **Prompt 模板**：与 `PROMPT_TEMPLATE_ENCODE_PREFIX/SUFFIX` 一致。
- **文本/图像 position 分离**：图像从 512 起，与 `tokenizer_max_length` 一致。
- **MROPE 与 attention_mask**：`forward_longcat` 行为对齐 Qwen2.5-VL 在 diffusers 中的用法。
- **Pack/unpack**：与 `pack_latents` / `unpack_latents` 逻辑一致。
- **Scheduler**：Flow Matching + Euler 离散、dynamic shifting、`calculate_shift` 与 `retrieve_timesteps` 对齐。
- **CFG**：公式 `neg + scale * (pos - neg)` 及可选的 renorm 与 diffusers 一致。

---

## 三、小结

- **原理**：LongCat-Image 在 xllm 中以 DiT 管线形式实现，采用 **Qwen2.5-VL 文本编码 + LongCatImageTransformer2DModel 去噪 + FlowMatchEulerDiscreteScheduler + VAE 解码**，通过多轴 RoPE、pack/unpack latent、CFG 等与 diffusers 对齐。
- **推理**：**Encode prompt → Prepare latents & position IDs → 缓存 RoPE → 多步 Denoising（Transformer + 可选 CFG + Scheduler step）→ Unpack → VAE decode → 后处理**。整条链路由 `LongCatImagePipeline::forward` 串联，通过 `DiTExecutor` / `DiTWorker` 接入 xllm 的 DiT 推理引擎。
