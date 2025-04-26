import torch
import time
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from utils import *
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--main-model-path",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
)
parser.add_argument(
    "--model-id",
    type=str,
    default="Qwen-1.5B",
)
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
    "--data-start",
    type=int,
)
parser.add_argument(
    "--data-end",
    type=int,
)

args = parser.parse_args()

assert args.model_id in args.main_model_path

if args.bench_name=="gsm8k":
    dataset = load_dataset("openai/gsm8k","main",split="test").to_list()[:500]
elif args.bench_name=="gaokao":
    dataset = load_dataset(
        "data/gaokao",
        split="train").to_list()
elif args.bench_name in ["math","olympiad","college_math"]:
    dataset=[]
    with open(f"data/{args.bench_name}.json","r") as f:
        for i in f.readlines():
            data = json.loads(i)
            dataset.append(data)

print(f"Evaluating on {args.bench_name}.")

if not args.data_start:
    args.data_start=0

if args.data_end:
    dataset=dataset[args.data_start:args.data_end]
else:
    dataset = dataset[args.data_start:]

model=AutoModelForCausalLM.from_pretrained(args.main_model_path,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(args.main_model_path)



correct=0
cur_correct=False


output_path=args.output_path+"/"+args.bench_name
if not os.path.exists(f"{output_path}"):
    os.makedirs(f"{output_path}")
output_path=output_path+"/base_"+args.model_id+".json"

for id,data in tqdm(enumerate(dataset)):
    torch.cuda.synchronize()
    start_time = time.time()

    messages = [
        {"role": "user", "content": data["question"]},
    ]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_dict = tokenizer(formatted_input, return_tensors="pt")

    torch.cuda.synchronize()
    time1 = time.time()
    output = model.generate(input_dict.input_ids.cuda(), max_new_tokens=20480, do_sample=False,
                            eos_token_id=tokenizer.convert_tokens_to_ids("</think>"))
    torch.cuda.synchronize()
    time2 = time.time()
    cot_len = output.size(1) - input_dict.input_ids.size(1)

    if output[0][-1].item() != tokenizer.convert_tokens_to_ids("</think>"):
        output=torch.cat([output,torch.tensor([tokenizer("</think>\n\n").input_ids[1:]],device=output.device)],dim=-1)


    output = model.generate(output, max_new_tokens=10240, do_sample=False, eos_token_id=tokenizer.eos_token_id)

    pred = tokenizer.decode(output.tolist()[0], skip_special_tokens=True)

    torch.cuda.synchronize()
    end_time = time.time()


    if args.bench_name in ["gaokao","college_math"]:
        with open(output_path,"a") as f:
            ans_json = {
                "id": id+args.data_start,
                "answer": data["answer"],
                "pred": pred,
                "cot_time": time2-time1,
                "cot_len": cot_len,
                "total_time": end_time-start_time,
            }
            f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")
    elif args.bench_name in ["gsm8k","olympiad"]:
        ground_truth_num = extract_answer(data["answer"])
        predicted_num = extract_answer(pred)
        if ground_truth_num is not None and predicted_num is not None and ground_truth_num == predicted_num:
            correct += 1
            cur_correct = True
        else:
            cur_correct = False
        with open(output_path,"a") as f:
            ans_json = {
                "id": id+args.data_start,
                "answer": data["answer"],
                "pred": pred,
                "cot_time": time2-time1,
                "cot_len": cot_len,
                "total_time": end_time-start_time,
            }
            f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")
    elif args.bench_name == "math":
        ground_truth_num = extract_answer(data["solution"])
        predicted_num = extract_answer(pred)
        if ground_truth_num is not None and predicted_num is not None and ground_truth_num == predicted_num:
            correct += 1
            cur_correct = True
        else:
            cur_correct = False
        with open(output_path,"a") as f:
            ans_json = {
                "id": id+args.data_start,
                "answer": data["answer"],
                "pred": pred,
                "cot_time": time2-time1,
                "cot_len": cot_len,
                "total_time": end_time-start_time,
            }
            f.write(json.dumps(ans_json,ensure_ascii=False) + "\n")


print(f"Complete {args.bench_name}!")
if args.bench_name in ["gsm8k","math","olympiad"]:
    print("acc:",round(correct/len(dataset),4))
elif args.bench_name in ["gaokao","college_math"]:
    cal_gaokao_score(output_path)