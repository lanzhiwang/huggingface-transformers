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

rm -rf src/transformers.egg-info/ src/transformers/__pycache__/ src/transformers/generation/__pycache__/ src/transformers/integrations/__pycache__/ src/transformers/loss/__pycache__/ src/transformers/models/__pycache__/ src/transformers/models/auto/__pycache__/ src/transformers/models/llama/__pycache__/ src/transformers/quantizers/__pycache__/ src/transformers/utils/__pycache__/

```
