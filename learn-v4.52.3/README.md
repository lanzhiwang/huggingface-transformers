```bash

docker build -t transformers:v1 -f Dockerfile ./

docker run \
-d -p 0.0.0.0:24:22 \
-v ~/work/code/go_code/ai/huggingface/transformers:/transformers \
-w /transformers \
--name transformers \
transformers:v1

pip install -e ./

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

find . -name "*.py" -exec black {} \;

```
