import torch
import time
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k","main",split="test").to_list()[500:1000]


draft_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


draft_model=AutoModelForCausalLM.from_pretrained(draft_model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                device_map="auto")

draft_tokenizer=AutoTokenizer.from_pretrained(draft_model_path)

def extract_number(answer):
    numbers = re.findall(r"-?\d+\.?\d*", answer)
    if numbers and numbers[-1].endswith("."):
        numbers = re.findall(r"-?\d+", answer)
    return numbers[-1] if numbers else None

def draft_cot(model, tokenizer, prompt, num_chain=3, temperature=0.6, max_length=1000):
    messages = [
        {"role": "user", "content": prompt},
    ]

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_dict = tokenizer(formatted_input, return_tensors="pt")
    input_ids = input_dict["input_ids"].repeat(num_chain, 1)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(num_chain, -1)
    with torch.no_grad():
        output = model(input_ids=input_ids.cuda(), position_ids=position_ids.cuda())
    past_key_values = output.past_key_values
    prompt_len = input_ids.size(1)
    logits = output.logits[:, -1, :]

    chains = [[] for _ in range(num_chain)]
    end = [False] * num_chain

    eos_token_id = tokenizer.convert_tokens_to_ids("</think>")


    for step in range(max_length):
        if all(end):
            break
        probs = torch.softmax(logits / temperature, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for i in range(num_chain):
            if not end[i]:
                chains[i].append(next_tokens[i].item())
                if next_tokens[i].item() == eos_token_id:
                    end[i] = True

        input_ids = next_tokens.unsqueeze(-1)
        position_ids = torch.full((num_chain,1),step+prompt_len,dtype=torch.int32)
        with torch.no_grad():
            output = model(input_ids=input_ids.cuda(), position_ids=position_ids.cuda(), past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]


    return chains

def pick_draft(prompt,chains,label):
    prompt="I will provide several Reasons for the Question. Please choose the best reasoning path and give the serial Number directly (you can only choose one). \n\nQuestion: " + prompt + "\n\nHere are the reasoning paths:\n\n"

    for i in range(len(chains)):
        if chains[i].endswith("\n</think>"):
            chains[i]=chains[i][:-len("\n</think>")]
        prompt+="Reason "+str(i+1)+": "+chains[i]+"\n\n"

    prompt += "Reason " + str(len(chains) + 1) + ": All reasoning paths above are wrong.\n\n"


    formatted_input=prompt+"<｜Assistant｜>"+"Number of the best reasoning path: "
    with open("/public/home/ljt/wjk/speculative_cot/train/data/gsm8k_500-1000_5chain.json", "a") as f:
        ans_json = {
            "input": formatted_input,
            "label": label,
        }
        f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")




for id,data in tqdm(enumerate(dataset)):
    chains = draft_cot(draft_model, draft_tokenizer, data["question"], num_chain=5, temperature=0.6, max_length=1000)
    draft = [draft_tokenizer.decode(chain, skip_special_tokens=True) for chain in chains]
    label=[]
    ground_truth_num = extract_answer(data["answer"])
    for i in range(len(draft)):
        predicted_num = extract_answer(draft[i])
        if predicted_num==ground_truth_num:
            label.append(i+1)
    if len(label)==0:
        label.append(6)

    pick_draft(data["question"],draft,label)
