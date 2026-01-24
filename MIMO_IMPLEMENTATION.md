# xllm 中 MiMo 模型实现原理与推理流程

## 一、实现原理

### 1.1 架构设计

MiMo 模型在 xllm 中的实现采用了**模板继承**的设计模式，基于 Qwen2 架构：

```cpp
class MiMoModelImpl : public LlmModelImplBase<layer::Qwen2DecoderLayer>
```

**核心设计思想：**
- **复用 Qwen2 架构**：MiMo 与 Qwen2 共享相同的 decoder layer 结构
- **模板化设计**：通过 `LlmModelImplBase<DecoderLayerType>` 模板类实现代码复用
- **模块化组件**：每个组件（embedding、attention、mlp、norm）都是独立的模块

### 1.2 模型结构

```
MiMoForCausalLM
├── MiMoModel (LlmModelImplBase<Qwen2DecoderLayer>)
│   ├── embed_tokens_ (WordEmbedding)
│   ├── layers_[0..N-1] (Qwen2DecoderLayer × N)
│   │   ├── input_norm_ (RMSNorm)
│   │   ├── self_attn_ (Qwen2Attention)
│   │   │   ├── qkv_proj_ (QKVParallelLinear)
│   │   │   ├── o_proj_ (RowParallelLinear)
│   │   │   └── rotary_emb_ (MRotaryEmbedding)
│   │   ├── post_norm_ (RMSNorm)
│   │   └── mlp_ (DenseMLP)
│   └── norm_ (RMSNorm)
└── lm_head_ (LmHead)
```

### 1.3 关键特性

#### 1.3.1 权重加载机制

xllm 实现了**自动权重格式转换**，兼容 HuggingFace 格式：

**Attention 权重转换：**
```cpp
// HuggingFace 格式: q_proj, k_proj, v_proj (分离)
// xllm 格式: qkv_proj (合并)
qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
```

**MLP 权重转换：**
```cpp
// HuggingFace 格式: gate_proj, up_proj (分离)
// xllm 格式: gate_up_proj (合并)
gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
```

这种设计的好处：
- **性能优化**：合并的权重矩阵可以使用融合 kernel，减少内存访问
- **兼容性**：自动处理 HuggingFace 格式，无需手动转换权重

#### 1.3.2 mRoPE 支持

MiMo 支持 **Multi-head Rotary Position Embedding (mRoPE)**：

```cpp
std::pair<torch::Tensor, torch::Tensor> apply_mrope(
    const torch::Tensor positions) override {
  // 1. 获取位置对应的 cos/sin 值
  auto target_cos_sin = cos_sin_.index({positions});
  
  // 2. 分割为 cos 和 sin
  auto target_cos_sin_chunks = target_cos_sin.chunk(2, -1);
  
  // 3. 应用 mRoPE 的特殊选择逻辑
  // 根据 mrope_section_ 配置选择不同的 head 维度
  // ...
}
```

mRoPE 允许不同的 attention head 使用不同的旋转频率，增强位置编码的表达能力。

#### 1.3.3 模型注册机制

xllm 使用**宏注册**机制实现模型自动发现：

```cpp
// 注册模型工厂
REGISTER_CAUSAL_MODEL(mimo, MiMoForCausalLM);

// 注册参数加载器
REGISTER_MODEL_ARGS(mimo, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mimo");
  LOAD_ARG_OR(vocab_size, "vocab_size", 151680);
  // ... 其他参数
});
```

**工作流程：**
1. 从 `config.json` 读取 `model_type: "mimo"`
2. 通过 `ModelRegistry` 查找对应的工厂函数
3. 创建 `MiMoForCausalLM` 实例
4. 使用注册的参数加载器解析配置

## 二、推理流程

### 2.1 整体流程

```
API Request
    ↓
LLMWorker::step()
    ↓
ModelExecutor::forward()
    ↓
MiMoForCausalLM::forward()
    ↓
MiMoModel::forward()
    ↓
Qwen2DecoderLayer::forward() × N layers
    ↓
MiMoForCausalLM::logits()
    ↓
Sampler::forward()
    ↓
Response
```

### 2.2 详细步骤

#### 步骤 1: Worker 层接收请求

```cpp
// llm_worker_impl.cpp
std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
  // 1. 准备输入：token_ids, positions, kv_caches
  // 2. 调用模型执行器
  auto hidden_states = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  
  // 3. 计算 logits
  torch::Tensor logits = model_->logits(hidden_states, ...);
  
  // 4. 采样
  SampleOutput sample_output = sampler_->forward(logits, ...);
  
  return output;
}
```

#### 步骤 2: 模型前向传播

**2.2.1 Embedding 层**

```cpp
// llm_model_base.h
torch::Tensor h;
if (inputs_embeds.defined()) {
  h = inputs_embeds;  // 使用预计算的 embedding
} else {
  h = embed_tokens_(tokens);  // Token ID → Embedding
}
// 输出: [num_tokens, hidden_size]
```

**2.2.2 构建 Attention Metadata**

```cpp
// 构建注意力计算的元数据
auto& attn_metadata = *(modified_input_params.attn_metadata);
// 包括: cu_seq_lens, block_tables, mask 等

// 处理 mRoPE
if (positions.dim() == 2) {
  std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) = 
      apply_mrope(positions);
}
```

**2.2.3 逐层前向传播**

```cpp
std::optional<torch::Tensor> residual;
for (size_t i = 0; i < layers_.size(); i++) {
  attn_metadata.plan_info->layer_id = i;
  auto& layer = layers_[i];
  
  // 每个 decoder layer 的前向传播
  h = layer(h, residual, positions, attn_metadata, 
            kv_caches[i], modified_input_params);
}
```

**Decoder Layer 内部流程：**

```cpp
// qwen2_decoder_layer.cpp
torch::Tensor Qwen2DecoderLayerImpl::forward(...) {
  // 1. Pre-attention normalization
  std::tie(x, residual) = input_norm_->forward(x, residual);
  
  // 2. Self-attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  
  // 3. Post-attention normalization
  std::tie(x, residual) = post_norm_->forward(x, residual);
  
  // 4. MLP
  x = mlp_->forward(x);
  
  return x;
}
```

**Attention 内部流程：**

```cpp
// qwen2_attention.cpp
torch::Tensor Qwen2AttentionImpl::forward(...) {
  // 1. QKV 投影 (融合 kernel)
  auto qkv = qkv_proj_->forward(hidden_states);
  
  // 2. 分割 Q, K, V
  auto q = qkv.slice(-1, 0, q_size_);
  auto k = qkv.slice(-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
  
  // 3. RoPE 位置编码
  rotary_emb_->forward(q, k, positions, attn_metadata);
  
  // 4. Attention 计算 + KV Cache 更新
  auto out = attn_->forward(attn_metadata, q, k, v, kv_cache);
  
  // 5. 输出投影
  return o_proj_->forward(out);
}
```

**2.2.4 最终归一化**

```cpp
// 最后一层归一化
return std::get<0>(norm_(h, residual));
// 输出: [num_tokens, hidden_size]
```

#### 步骤 3: 计算 Logits

```cpp
// llm_model_base.h
virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                             const torch::Tensor& selected_idxes) {
  auto h = hidden_states;
  if (selected_idxes.defined()) {
    h = h.index_select(0, selected_idxes);  // 选择特定 token
  }
  return lm_head_(h);  // [num_tokens, vocab_size]
}
```

#### 步骤 4: 采样

```cpp
// 根据 logits 采样下一个 token
SampleOutput sample_output = sampler_->forward(logits, sampling_params);
// 支持: greedy, top-k, top-p, temperature 等采样策略
```

### 2.3 关键优化技术

#### 2.3.1 KV Cache

- **目的**：避免重复计算历史 token 的 K/V
- **实现**：每个 layer 维护独立的 `KVCache`
- **更新**：每次 decode 只计算新 token 的 K/V，与缓存拼接

#### 2.3.2 并行计算

- **Tensor Parallelism (TP)**：模型权重分片到多个设备
- **Data Parallelism (DP)**：不同序列分配到不同设备
- **Pipeline Parallelism (PP)**：不同层分配到不同设备

#### 2.3.3 融合 Kernel

- **QKV 融合**：一次矩阵乘法同时计算 Q、K、V
- **Gate-Up 融合**：MLP 的 gate 和 up projection 合并计算
- **Flash Attention**：优化的注意力计算 kernel

#### 2.3.4 内存优化

- **Residual 复用**：使用 `std::optional<torch::Tensor>` 避免不必要的拷贝
- **In-place 操作**：尽可能使用原地操作减少内存分配
- **量化支持**：支持 INT8、INT4 等量化格式

## 三、与 Qwen2 的关系

### 3.1 相同点

- **架构一致**：使用相同的 `Qwen2DecoderLayer`
- **组件相同**：Attention、MLP、Norm 的实现完全一致
- **权重兼容**：可以直接加载 Qwen2 格式的权重（通过自动转换）

### 3.2 不同点

- **模型类型**：`model_type: "mimo"` vs `"qwen2"`
- **参数配置**：MiMo 有特定的默认参数（如 `rope_theta: 640000`）
- **mRoPE 支持**：MiMo 实现了 `apply_mrope` 方法（Qwen2 可能不使用）

### 3.3 设计优势

1. **代码复用**：最大化利用 Qwen2 的成熟实现
2. **维护简单**：Qwen2 的优化自动应用到 MiMo
3. **兼容性好**：支持 HuggingFace 格式，无需额外转换工具

## 四、总结

xllm 实现 MiMo 的核心思想是**基于 Qwen2 架构的模板化设计**：

1. **架构层面**：继承 `LlmModelImplBase<Qwen2DecoderLayer>`，复用所有底层实现
2. **权重层面**：自动转换 HuggingFace 格式到 xllm 优化格式
3. **推理层面**：标准的 Transformer decoder 前向传播流程
4. **优化层面**：享受 xllm 的所有性能优化（KV Cache、融合 kernel、并行等）

这种设计既保证了实现的正确性，又充分利用了 xllm 的优化能力，是一个高效且可维护的实现方案。
