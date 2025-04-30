# Efficient Reasoning for LLMs through Speculative Chain-of-Thought

## Installation
```bash
pip install -r requirements.txt
```

## Evaluate the Base Reasoning Models on Benchmarks
```bash
python base.py --bench-name gsm8k --main-model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --model-id Qwen-32B
```

## Evaluate the Performance of SCoT
```bash
python speculative_cot.py --bench-name gsm8k
```


## To Train Your Own LoRA Weights for SCoT 

We provide datasets for training the target model (32B) and the draft model (1.5B) in the ./train/data and ./train_draft/data directories respectively.
We also provide the trained lora weights (32B and 1.5B).
In case you need to make a custom dataset or retrain the lora module, please refer to the following script.

### Prepare data and train the target model
```bash
cd train
python prepare_data.py
python lora_train.py
```
### Prepare data and train the draft model
```bash
cd train_draft
python prepare_data.py
python lora_train.py
```
