from transformers import LlamaConfig, LlamaModel
import torch

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096 // 2,
    intermediate_size=11008 // 2,
    num_hidden_layers=32 // 2,
    num_attention_heads=32 // 2,
    max_position_embeddings=2048 // 2,
)
print(config)

model = LlamaModel(config=config)
print(model)

inputs_ids = torch.randint(0, config.vocab_size, (4, 30), device=torch.device('cpu'))
print(inputs_ids)

# res = model(inputs_ids)
# print(res)
