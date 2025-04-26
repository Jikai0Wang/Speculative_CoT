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
