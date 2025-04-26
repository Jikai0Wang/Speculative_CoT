import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
os.environ["WANDB_MODE"] = "disabled"

class ReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=30000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = self.tokenizer(sample['pred'], return_tensors='pt')
        # label = sample['label']
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            # 'attention_mask': inputs['attention_mask'].squeeze(0),
            # 'labels': label
        }


def load_data(tokenizer):
    data=[]
    with open("data/gsm8k-D-Qwen-1500.json") as f:
        for i in f.readlines():
            d=json.loads(i)
    return ReasoningDataset(data, tokenizer)


tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')  # 替换为你的模型

model = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    torch_dtype=torch.float16,
    device_map='auto'
)
# model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.1,
    bias='none'
)
model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if 'lora' in name:
        param.data = param.data.to(torch.float32)
        param.requires_grad = True

train_dataset = load_data(tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)



def compute_loss(model, inputs):
    outputs = model(inputs.input_ids)
    index = (inputs.input_ids[0] == 151648).nonzero().item()
    label=inputs.input_ids[0,index+1:]
    logits = outputs.logits[0, index:-1, :]

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, label.to(logits.device))


    return loss,outputs

training_args = TrainingArguments(
    # gradient_accumulation_steps=4,
    output_dir='./lora_model',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=1):
        loss,outputs = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()

model.save_pretrained('./lora_model')