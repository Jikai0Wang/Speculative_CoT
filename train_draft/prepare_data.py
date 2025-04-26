import torch
import time
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from utils import *

dataset = load_dataset("openai/gsm8k","main",split="test").to_list()[500:1000]


main_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


model=AutoModelForCausalLM.from_pretrained(main_model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(main_model_path)



for id,data in tqdm(enumerate(dataset)):

    messages = [
        {"role": "user", "content": data["question"]},
    ]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_dict = tokenizer(formatted_input, return_tensors="pt")


    output = model.generate(input_dict.input_ids.cuda(), max_new_tokens=20480, do_sample=False, eos_token_id=tokenizer.convert_tokens_to_ids("</think>"))

    pred = tokenizer.decode(output.tolist()[0], skip_special_tokens=False)



    with open("data/gsm8k-test-500-1000","a") as f:
        ans_json = {
            "pred": pred,
        }
        f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")