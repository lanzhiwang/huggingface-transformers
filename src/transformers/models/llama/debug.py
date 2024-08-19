from transformers import LlamaModel, LlamaConfig
import torch

configuration = LlamaConfig(vocab_size=32000 // 8,
                            hidden_size=4096 // 8,
                            intermediate_size=11008 // 8,
                            num_hidden_layers=32 // 8,
                            num_attention_heads=32 // 8,
                            num_key_value_heads=None,
                            hidden_act="silu",
                            max_position_embeddings=2048 // 8,
                            initializer_range=0.02,
                            rms_norm_eps=1e-6,
                            use_cache=True,
                            pad_token_id=None,
                            bos_token_id=1,
                            eos_token_id=2,
                            pretraining_tp=1,
                            tie_word_embeddings=False,
                            rope_theta=10.0,
                            rope_scaling=None,
                            attention_bias=False,
                            attention_dropout=0.0)
# print(configuration)
# LlamaConfig {
#   "attention_bias": false,
#   "attention_dropout": 0.0,
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 512,
#   "initializer_range": 0.02,
#   "intermediate_size": 1376,
#   "max_position_embeddings": 256,
#   "model_type": "llama",
#   "num_attention_heads": 4,
#   "num_hidden_layers": 4,
#   "num_key_value_heads": 4,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-06,
#   "rope_scaling": null,
#   "rope_theta": 10.0,
#   "tie_word_embeddings": false,
#   "transformers_version": "4.38.2",
#   "use_cache": true,
#   "vocab_size": 4000  # 词表大小
# }

# Initializing a model from the llama-7b style configuration
model = LlamaModel(config=configuration)

# # Accessing the model configuration
# configuration = model.config
# print(configuration)

input_ids = torch.randint(low=0, high=configuration.vocab_size, size=(4, 10))
# print(input_ids)
# tensor([[2361, 2866, 2923, 1729, 1193, 1706, 1675, 1310, 2619, 1662],
#         [ 561, 1175, 3414, 3245, 2915, 1947, 1735, 3996,  483, 3318],
#         [2257, 1932, 1891, 1875, 2119, 1932, 1076, 2171, 2220,   35],
#         [3479, 3721, 3889, 3363,  973,  339, 3301, 3628,   28, 1371]])
# print(input_ids.shape)
# torch.Size([4, 10])

res = model(input_ids)
# print(res)
