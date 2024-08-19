<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, ALBERT, BERT, DistilBERT, RoBERTa, XLNet...
GPT and GPT-2 are trained or fine-tuned using a **causal language modeling (CLM)** loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a **masked language modeling (MLM)** loss. XLNet uses **permutation language modeling (PLM)**,
you can find more information about the differences between those objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).
微调（或从头开始训练）用于在 GPT、GPT-2、ALBERT、BERT、DistilBERT、RoBERTa、XLNet 的文本数据集上进行语言建模的库模型...
GPT 和 GPT-2 使用以下方法进行训练或微调因果语言模型 (CLM) 损失，
而 ALBERT、BERT、DistilBERT 和 RoBERTa 使用掩码语言模型 (MLM) 损失进行训练或微调。
XLNet 使用排列语言建模 (PLM)，
您可以在我们的模型摘要中找到有关这些目标之间差异的更多信息。

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.
提供了两组脚本。第一组利用 Trainer API。后缀中带有no_trainer的第二组使用自定义训练循环并利用 🤗 Accelerate 库。两个集合都使用 🤗 数据集库。如果您需要对数据集进行额外处理，您可以根据需要轻松自定义它们。

**Note:** The old script `run_language_modeling.py` is still available [here](https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py).
注意：旧脚本run_language_modeling.py仍然可以在此处使用。

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.
以下示例将在我们中心托管的数据集上运行，或使用您自己的文本文件进行训练和验证。我们在下面给出了两者的例子。

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.
以下示例在 WikiText-2 上微调 GPT-2。我们使用原始的 WikiText-2（在标记化之前没有替换任何标记）。这里的损失是因果语言建模的损失。

```bash
python run_clm.py \
    --model_name_or_path openai-community/gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.
在单个 K80 GPU 上训练大约需要半小时，运行评估大约需要一分钟。一旦对数据集进行微调，它的困惑度分数就会达到约 20。

To run on your own training and validation files, use the following command:
要在您自己的训练和验证文件上运行，请使用以下命令：

```bash
python run_clm.py \
    --model_name_or_path openai-community/gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_clm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:
这使用内置的 HuggingFace Trainer进行训练。如果您想使用自定义训练循环，您可以利用或改编run_clm_no_trainer.py脚本。查看脚本以获取支持的参数列表。示例如下所示：

```bash
python run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /tmp/test-clm
```

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.
以下示例对 WikiText-2 上的 RoBERTa 进行微调。在这里，我们也使用原始的 WikiText-2。由于BERT/RoBERTa具有双向机制，损失不同；因此，我们使用与预训练期间使用的相同损失：掩码语言建模。

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).
根据 RoBERTa 论文，我们使用动态掩码而不是静态掩码。因此，模型的收敛速度可能会稍慢（过度拟合需要更多的时间）。

```bash
python run_mlm.py \
    --model_name_or_path FacebookAI/roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

To run on your own training and validation files, use the following command:
要在您自己的训练和验证文件上运行，请使用以下命令：

```bash
python run_mlm.py \
    --model_name_or_path FacebookAI/roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).
如果您的数据集按每行一个样本进行组织，则可以使用--line_by_line标志（否则脚本会连接所有文本，然后将它们拆分为相同长度的块）。

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_mlm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:
这使用内置的 HuggingFace Trainer进行训练。如果您想使用自定义训练循环，您可以利用或改编run_mlm_no_trainer.py脚本。查看脚本以获取支持的参数列表。示例如下所示：

```bash
python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path FacebookAI/roberta-base \
    --output_dir /tmp/test-mlm
```

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.
注意：在 TPU 上，您应该将--pad_to_max_length标志与--line_by_line标志结合使用，以确保所有批次具有相同的长度。

### Whole word masking

This part was moved to `examples/research_projects/mlm_wwm`.

### XLNet and permutation language modeling

XLNet uses a different training objective, which is permutation language modeling. It is an autoregressive method
to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input
sequence factorization order.
XLNet 使用不同的训练目标，即排列语言建模。它是一种自回归方法，通过最大化输入序列分解顺序的所有排列的预期似然来学习双向上下文。

We use the `--plm_probability` flag to define the ratio of length of a span of masked tokens to surrounding
context length for permutation language modeling.
我们使用--plm_probability标志来定义屏蔽标记范围的长度与周围上下文长度的比率，以进行排列语言建模。

The `--max_span_length` flag may also be used to limit the length of a span of masked tokens used
for permutation language modeling.
--max_span_length标志还可用于限制用于排列语言建模的屏蔽标记的跨度的长度。

Here is how to fine-tune XLNet on wikitext-2:
以下是如何在 wikitext-2 上微调 XLNet：

```bash
python run_plm.py \
    --model_name_or_path=xlnet/xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

To fine-tune it on your own training and validation file, run:
要在您自己的训练和验证文件上对其进行微调，请运行：

```bash
python run_plm.py \
    --model_name_or_path=xlnet/xlnet-base-cased \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).
如果您的数据集按每行一个样本进行组织，则可以使用--line_by_line标志（否则脚本会连接所有文本，然后将它们拆分为相同长度的块）。

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.
注意：在 TPU 上，您应该将--pad_to_max_length标志与--line_by_line标志结合使用，以确保所有批次具有相同的长度。

## Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` to the command line. This is currently supported by `run_mlm.py` and `run_clm.py`.
要使用对大型数据集非常有用的流数据集模式，请将--streaming添加到命令行。目前run_mlm.py和run_clm.py支持此功能。

## Low Cpu Memory Usage

To use low cpu memory mode which can be very useful for LLM, add `--low_cpu_mem_usage` to the command line. This is currently supported by `run_clm.py`,`run_mlm.py`, `run_plm.py`,`run_mlm_no_trainer.py` and `run_clm_no_trainer.py`.
从头开始训练模型时，可以在--config_overrides的帮助下覆盖配置值：

## Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_clm.py --model_type openai-community/gpt2 --tokenizer_name openai-community/gpt2 \ --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
[...]
```

This feature is only available in `run_clm.py`, `run_plm.py` and `run_mlm.py`.
此功能仅在run_clm.py 、 run_plm.py和run_mlm.py中可用。
