from transformers import LlamaConfig, LlamaModel
import torch

config = LlamaConfig(
    vocab_size=5,
    hidden_size=4096 // 1024,  # 4
    intermediate_size=11008 // 1834,  # 6
    num_hidden_layers=32 // 8,  # 4
    num_attention_heads=32 // 8,  # 4
    max_position_embeddings=2048 // 526,  # 3
)
print(config)

model = LlamaModel(config=config)
print(model)
print(model.layers[0])
"""
LlamaModel(
  (embed_tokens): Embedding(5, 4)
  (layers): ModuleList(
    (0-3): 4 x LlamaDecoderLayer(
      (self_attn): LlamaAttention(
        (q_proj): Linear(in_features=4, out_features=4, bias=False)
        (k_proj): Linear(in_features=4, out_features=4, bias=False)
        (v_proj): Linear(in_features=4, out_features=4, bias=False)
        (o_proj): Linear(in_features=4, out_features=4, bias=False)
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=4, out_features=6, bias=False)
        (up_proj): Linear(in_features=4, out_features=6, bias=False)
        (down_proj): Linear(in_features=6, out_features=4, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): LlamaRMSNorm((4,), eps=1e-06)
      (post_attention_layernorm): LlamaRMSNorm((4,), eps=1e-06)
    )
  )
  (norm): LlamaRMSNorm((4,), eps=1e-06)
  (rotary_emb): LlamaRotaryEmbedding()
)

LlamaDecoderLayer(
  (self_attn): LlamaAttention(
    (q_proj): Linear(in_features=4, out_features=4, bias=False)
    (k_proj): Linear(in_features=4, out_features=4, bias=False)
    (v_proj): Linear(in_features=4, out_features=4, bias=False)
    (o_proj): Linear(in_features=4, out_features=4, bias=False)
  )
  (mlp): LlamaMLP(
    (gate_proj): Linear(in_features=4, out_features=6, bias=False)
    (up_proj): Linear(in_features=4, out_features=6, bias=False)
    (down_proj): Linear(in_features=6, out_features=4, bias=False)
    (act_fn): SiLU()
  )
  (input_layernorm): LlamaRMSNorm((4,), eps=1e-06)
  (post_attention_layernorm): LlamaRMSNorm((4,), eps=1e-06)
)
"""

inputs_ids = torch.randint(0, config.vocab_size, (4, 3), device=torch.device('cpu'))
print(inputs_ids)

res = model(inputs_ids)
print("res:", res)
print("res:", res.last_hidden_state.size())  # res: torch.Size([4, 3, 4])
