```bash
docker run -ti --rm \
-v ~/work/code/go_code/ai/huggingface/transformers:/transformers \
-w /transformers \
docker-mirrors.alauda.cn/library/python:3.10.12-bullseye \
bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple yapf

python -m venv .env
source .env/bin/activate
pip install -e .
pip install 'transformers[torch]'
pip install 'transformers[testing]'
pip install 'transformers[dev]'


>>> from transformers import pipeline
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")

```
