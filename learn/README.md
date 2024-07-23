```bash
docker run -ti --rm \
-v ~/work/code/go_code/ai/huggingface/transformers:/transformers \
-w /transformers \
docker-mirrors.alauda.cn/library/python:3.10.12-bullseye \
bash

python -m venv .env
source .env/bin/activate
pip install -e .
pip install 'transformers[torch]'

>>> from transformers import pipeline
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")


```
