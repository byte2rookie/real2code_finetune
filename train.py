from datasets import load_dataset
import json
import random


def read_json(file_path):
    """读取 JSON 文件"""
    with open(file_path, "r") as f:
        return json.load(f)


# 定义模式列表
modes = ["absolute", "obb_rel", "obb_rot"]
# modes=['absolute']
# 假设 train.json 在当前目录，可根据实际路径调整
train_json_path = "train.json"
test_json_path = "test.json"
# 读取训练数据 JSON
train_data_json = read_json(train_json_path)
test_data_json = read_json(test_json_path)

for mode in modes:
    all_train_entries = []
    all_test_entries = []
    # 遍历每个物体 ID 和对应的信息
    for obj_id, obj_info in train_data_json.items():
        mode_info = obj_info.get(mode, {})
        bbox_code = mode_info.get("bbox_code", "")
        label_code = mode_info.get("label_code", "")
        # 确保 bbox_code 和 label_code 存在
        if bbox_code and label_code:
            all_train_entries.append((bbox_code, label_code))

    # 随机打乱数据
    random.shuffle(all_train_entries)
    train_num = int(len(all_train_entries))

    # 遍历每个物体 ID 和对应的信息
    for obj_id, obj_info in test_data_json.items():
        mode_info = obj_info.get(mode, {})
        bbox_code = mode_info.get("bbox_code", "")
        label_code = mode_info.get("label_code", "")
        # 确保 bbox_code 和 label_code 存在
        if bbox_code and label_code:
            all_test_entries.append((bbox_code, label_code))

    # 随机打乱数据
    random.shuffle(all_test_entries)
    test_num = int(len(all_test_entries))

    # 写入训练集 JSONL 文件
    with open(f"{mode}_train_data.jsonl", "w") as f:
        for bbox, label in all_train_entries[:train_num]:
            entry = {"context": "", "question": bbox, "answer": label}
            f.write(json.dumps(entry) + "\n")

    # 写入验证集 JSONL 文件
    with open(f"{mode}_test_data.jsonl", "w") as f:
        for bbox, label in all_test_entries[:test_num]:
            entry = {"context": "", "question": bbox, "answer": label}
            f.write(json.dumps(entry) + "\n")

    print(
        f"Mode: {mode}, Train entries: {len(all_train_entries[:train_num])}, Test entries: {len(all_test_entries[:test_num])}"
    )

from datetime import datetime
import os
import sys
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# 加载自己的数据集
from datasets import load_dataset

train_dataset = load_dataset(
    "json", data_files="obb_rel_train_data.jsonl", split="train"
)
test_dataset = load_dataset("json", data_files="obb_rel_test_data.jsonl", split="train")

# 读取模型
base_model = "/data/winter25/zhouzy/ZZY/real2code/Scaled_Dataset/model"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=800,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""You are an AI assistant trained to understand 3D scenes and object relationships. Given the following Oriented Bounding Box (OBB) information, your task is to generate a list of child joints that describes the articulations between object parts.

OBB Information:
### Input:
{data_point["question"]}

Generate a number of root_geom,which means the base object,relative to OBB ID
- root_geom: Integer relative to/ selected from  input OBB ID
Generate a list of child joints. Each joint should be described by a dictionary with the following keys:
- box: The ID of the child bounding box
- type: The joint type ('hinge' for revolute joints, 'slide' for prismatic joints)
- idx: The rotation axis index (0 for x-axis, 1 for y-axis, 2 for z-axis)
- edge: Edge coordinates on the OBB, for example [1, -1]
- sign: Direction of the joint (+1 or -1)

IMPORTANT: Your response must contain ONLY the root_geom number and child_joints list, exactly as shown below, with no additional text before or after:

root_geom=[root_geom_number] 
child_joints = [
    dict(box=[child OBB ID], type=[joint type], idx=[rotation axis index], edge=[edge coordinates], sign=[direction]),
    # Additional joints as needed
]


Generate the geom_number and child_joints list:

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_test_dataset = test_dataset.map(generate_and_tokenize_prompt)


model.train()  # put model back into training mode
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)


# keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True


batch_size = 32
per_device_train_batch_size = 32
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "code-llama-ft"

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=100,
    max_steps=400,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    eval_strategy="steps",  # if val_set_size > 0 else "no",
    save_strategy="steps",
    eval_steps=20,
    save_steps=20,
    output_dir=output_dir,
    load_best_model_at_end=False,
    group_by_length=True,  # group sequences of roughly the same length together to speed up training
    report_to="none",  # if use_wandb else "none", wandb
    run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # if use_wandb else None,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)


trainer.train()
