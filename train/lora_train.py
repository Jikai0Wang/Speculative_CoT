import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
os.environ["WANDB_MODE"] = "disabled"

class ReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=8000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = self.tokenizer(sample['input'], return_tensors='pt')
        label = sample['label']
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'labels': label
        }


def load_data(tokenizer):
    data=[]
    with open("data/D-Qwen-gsm8k-5chain.json") as f:
        for i in f.readlines():
            data.append(json.loads(i))
    return ReasoningDataset(data, tokenizer)


tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')

model = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    torch_dtype=torch.float16,
    device_map='auto'
)


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

tokenizer_map={}
for i in range(1,7):
    tokenizer_map[i]=tokenizer.convert_tokens_to_ids(str(i))

def compute_loss(model, inputs, tokenizer_map):
    outputs = model(inputs.input_ids)
    logits = outputs.logits[:, -1, :]
    labels = [torch.tensor([tokenizer_map[num]]) for num in inputs['labels'].tolist()[0]]

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []

    min_idx=0
    min_val=99999
    for i in range(len(labels)):
        cur_loss=loss_fn(logits, labels[i].to(logits.device))
        losses.append(cur_loss)
        val=cur_loss[0].item()
        if val<min_val:
            min_val=val
            min_idx=i
    loss=losses[min_idx].squeeze(0)


    return loss,outputs

training_args = TrainingArguments(
    # gradient_accumulation_steps=4,
    output_dir='./lora_model',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=2,
    logging_steps=10,
    # fp16=True,
    learning_rate=1e-4,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=1):
        loss,outputs = compute_loss(model, inputs, tokenizer_map)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()

model.save_pretrained('./lora_model')