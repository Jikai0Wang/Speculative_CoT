import torch
import time
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from utils import *
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--main-model-path",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)
parser.add_argument("--draft-model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
parser.add_argument("--lora-model-path", type=str, default="train/lora_model")
parser.add_argument(
    "--bench-name",
    type=str,
    default="gsm8k",
    help="The name of the benchmark question set.",
)
parser.add_argument(
    "--output-path",
    type=str,
    default="output",
)
parser.add_argument(
    "--draft-output-file",
    type=str,
)
parser.add_argument(
    "--data-start",
    type=int,
)
parser.add_argument(
    "--data-end",
    type=int,
)
parser.add_argument(
    "--num-chains",
    type=int,
    default=5
)


args = parser.parse_args()

assert (args.draft_output_file is None or args.bench_name in args.draft_output_file)

if args.bench_name=="gsm8k":
    dataset = load_dataset("openai/gsm8k","main",split="test").to_list()[:500]
elif args.bench_name=="gaokao":
    dataset = load_dataset(
        "data/gaokao",
        split="train").to_list()
elif args.bench_name in ["math","aime24","olympiad","college_math"]:
    dataset = []
    with open(f"data/{args.bench_name}.json", "r") as f:
        for i in f.readlines():
            data = json.loads(i)
            dataset.append(data)

if not args.data_start:
    args.data_start=0

if args.data_end:
    dataset=dataset[args.data_start:args.data_end]
else:
    dataset = dataset[args.data_start:]

print(f"Evaluating on {args.bench_name}.")


draft_model=AutoModelForCausalLM.from_pretrained(args.draft_model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                device_map="cuda:0")
draft_tokenizer=AutoTokenizer.from_pretrained(args.draft_model_path)
draft_model=PeftModel.from_pretrained(draft_model, "train_draft/lora_model")
draft_model = draft_model.merge_and_unload()

main_model=AutoModelForCausalLM.from_pretrained(args.main_model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                device_map="auto")
main_tokenizer=AutoTokenizer.from_pretrained(args.main_model_path)
peft_model=AutoModelForCausalLM.from_pretrained(args.main_model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                device_map="auto")
peft_model = PeftModel.from_pretrained(peft_model, args.lora_model_path)
peft_model = peft_model.merge_and_unload()



def draft_cot(model, tokenizer, prompt, num_chain=3, temperature=0.6, max_length=4096):
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

def pick_draft(model,tokenizer,prompt,chains):
    prompt="I will provide several Reasons for the Question. Please choose the best reasoning path and give the serial Number directly (you can only choose one). \n\nQuestion: " + prompt + "\n\nHere are the reasoning paths:\n\n"

    for i in range(len(chains)):
        if chains[i].endswith("\n</think>"):
            chains[i]=chains[i][:-len("\n</think>")]
        prompt+="Reason "+str(i+1)+": "+chains[i]+"\n\n"

    prompt += "Reason " + str(len(chains) + 1) + ": All reasoning paths above are wrong.\n\n"

    formatted_input=prompt+"<｜Assistant｜>"+"Number of the best reasoning path: "

    tokenizer.pad_token = tokenizer.eos_token
    input_dict = tokenizer(formatted_input, return_tensors="pt")

    with torch.no_grad():
        output = model(input_dict.input_ids).logits[:, -1, :]
    pred = int(tokenizer.decode(torch.argmax(output, dim=-1)))

    if pred not in range(1,args.num_chains+2):
        _,indices=torch.topk(output,k=5, dim=-1)
        idx=1
        while pred not in range(1,args.num_chains+2):
            pred=int(tokenizer.decode(indices[0][idx]))
            idx+=1

    return pred-1


def get_output(model,tokenizer,prompt,draft):

    messages = [
        {"role": "user", "content": prompt},
    ]

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    formatted_input+=draft
    formatted_input+="\n</think>\n\n"

    input_dict = tokenizer(formatted_input,return_tensors="pt")

    output=model.generate(input_dict.input_ids.cuda(),max_new_tokens=10240,do_sample=False)
    lst=[]
    for ids in output:
        lst.append(tokenizer.decode(ids.tolist(),skip_special_tokens=True))
    return lst

def continue_reasoning_and_output(model,tokenizer,prompt,draft):

    messages = [
        {"role": "user", "content": prompt},
    ]

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_dict = tokenizer(formatted_input,return_tensors="pt")

    torch.cuda.synchronize()
    time1 = time.time()

    output=model.generate(input_dict.input_ids.cuda(),max_new_tokens=20480,do_sample=False,eos_token_id=tokenizer.convert_tokens_to_ids("</think>"))

    torch.cuda.synchronize()
    time2 = time.time()
    cot_len=output.size(1)-input_dict.input_ids.size(1)

    if output[0][-1].item() != tokenizer.convert_tokens_to_ids("</think>"):
        output=torch.cat([output,torch.tensor([tokenizer("</think>\n\n").input_ids[1:]],device=output.device)],dim=-1)

    output = model.generate(output, max_new_tokens=10240, do_sample=False,
                            eos_token_id=tokenizer.eos_token_id)

    lst=[]
    for ids in output:
        lst.append(tokenizer.decode(ids.tolist(),skip_special_tokens=True))

    return lst,time2-time1,cot_len

correct=0
cur_correct=False

all_drafts=[]
if args.draft_output_file:
    with open(args.draft_output_file,"r") as f:
        for i in f.readlines():
            res=json.loads(i)
            all_drafts.append(res["draft"])
    assert len(all_drafts) == len(dataset)

output_path=args.output_path+"/"+args.bench_name
if not os.path.exists(f"{output_path}"):
    os.makedirs(f"{output_path}")
output_path=output_path+"/sc.json"

for id,data in tqdm(enumerate(dataset)):
    torch.cuda.synchronize()
    start_time = time.time()

    if args.draft_output_file:
        draft=all_drafts[id]
    else:
        chains = draft_cot(draft_model, draft_tokenizer, data["question"], num_chain=args.num_chains, temperature=0.6, max_length=5000)
        draft = [draft_tokenizer.decode(chain, skip_special_tokens=True) for chain in chains]

    torch.cuda.synchronize()
    cur_time1 = time.time()

    best_draft_id=pick_draft(peft_model,main_tokenizer,data["question"],draft)

    torch.cuda.synchronize()
    cur_time2 = time.time()

    main_cot_len=0
    main_think_time=0
    if best_draft_id==len(draft):
        output,main_think_time,main_cot_len = continue_reasoning_and_output(main_model, main_tokenizer, data["question"], draft[0])
    else:
        output=get_output(main_model,main_tokenizer,data["question"],draft[best_draft_id])

    torch.cuda.synchronize()
    end_time = time.time()


    if args.draft_output_file:
        draft_len=0
        mean_draft_len=0
    else:
        draft_len_lst=[len(chain) for chain in chains]
        draft_len = max(draft_len_lst)
        mean_draft_len=sum(draft_len_lst) / len(draft_len_lst)

    if args.bench_name in ["gaokao","college_math"]:
        with open(output_path,"a") as f:
            ans_json = {
                "id": id+args.data_start,
                "draft": draft,
                "best_draft_id":best_draft_id,
                "answer": data["answer"],
                "pred": output[0],
                "draft_len": draft_len,
                "mean_draft_len": mean_draft_len,
                "total_time": end_time-start_time,
                "draft_time": cur_time1-start_time,
                "pick_time": cur_time2-cur_time1,
                "main_cot_len": main_cot_len,
                "main_think_time": main_think_time,
            }
            f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")
    elif args.bench_name=="math":
        ground_truth_num = extract_answer(data["solution"])
        predicted_num = extract_answer(output[0])
        if ground_truth_num is not None and predicted_num is not None and ground_truth_num == predicted_num:
            correct += 1
            cur_correct = True
        else:
            cur_correct = False
        with open(output_path,"a") as f:
            ans_json = {
                "id": id+args.data_start,
                "draft": draft,
                "best_draft_id": best_draft_id,
                "answer": data["answer"],
                "pred": output[0],
                "total_time": end_time-start_time,
                "correct": cur_correct,
                "draft_len": draft_len,
                "mean_draft_len": mean_draft_len,
                "draft_time": cur_time1 - start_time,
                "pick_time": cur_time2 - cur_time1,
                "main_cot_len": main_cot_len,
                "main_think_time": main_think_time,
            }
            f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")
    elif args.bench_name in ["gsm8k","olympiad"]:
        ground_truth_num = extract_answer(data["answer"])
        predicted_num = extract_answer(output[0])
        if ground_truth_num is not None and predicted_num is not None and ground_truth_num == predicted_num:
            correct += 1
            cur_correct = True
        else:
            cur_correct = False
        with open(output_path,"a") as f:
            ans_json = {
                "id": id+args.data_start,
                "draft": draft,
                "best_draft_id": best_draft_id,
                "answer": data["answer"],
                "pred": output[0],
                "total_time": end_time-start_time,
                "correct": cur_correct,
                "draft_len": draft_len,
                "mean_draft_len": mean_draft_len,
                "draft_time": cur_time1 - start_time,
                "pick_time": cur_time2 - cur_time1,
                "main_cot_len": main_cot_len,
                "main_think_time": main_think_time,
            }
            f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")

print(f"Complete {args.bench_name}!")
if args.bench_name in ["gsm8k","math","olympiad"]:
    print("acc:",round(correct/len(dataset),4))
elif args.bench_name in ["gaokao","college_math"]:
    cal_gaokao_score(output_path)