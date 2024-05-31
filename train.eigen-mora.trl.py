import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# from ordereduuid import OrderedUUID

# uuid = OrderedUUID()
# print("UID:", uuid)

# load jsonl dataset
# Load the dataset
ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")


# Define a function to transform the dataset format
def transform(example):
    return {"prompt": example["instruction"], "completion": example["output"]}


# Apply the transformation to the entire dataset
dataset = ds.map(transform)
print(dataset)

MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
MODEL_NAME = MODEL_ID.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{MODEL_NAME}-koalpaca-v1.1a",
    do_train=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    # weight_decay=0.1,
    num_train_epochs=2,
    warmup_ratio=0.03,
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    # fp16=True,
    tf32=True,
    optim="adafactor",
    # report_to="wandb",
    # gradient_checkpointing=True,
    neftune_noise_alpha=5,
    # fsdp=True,
)

max_position_embeddings = 1024

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    # torch_dtype=torch.bfloat16,
    device_map='auto',
    max_position_embeddings=max_position_embeddings,
    trust_remote_code=True,
)

peft_config = LoraConfig(
    # enable MoRA
    use_eigenmora=True,
    # type 1 (Sharing) for large lora ranks, Eq. 6 in paper
    # type 6 (RoPE based) for small lora ranks, Eq. 9 in paper
    # mora_type=6,
    # lora rank here, we will calculate corresponding $\hat{r}$ in MoRA
    r=1024,  # r hat
    # MoRA does not use lora_alpha
    # lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj"], #"all-linear",
    # target_modules="all-linear",
    # target_modules=["v_proj"], #"all-linear",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    # modules_to_save=['eigenmora_eigenvector_matrices'],
)
print(peft_config)

before_time = time.time()
model = get_peft_model(model, peft_config)
# model.save_pretrained('./peftmodel')
print(f"{time.time() - before_time}s for init model")
print(model)
model.print_trainable_parameters()
# model = AutoModelForCausalLM.from_pretrained('./peftmodel', device_map='auto')


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id


# if not tokenizer.pad_token:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
# if not tokenizer.chat_template:
#     print("Setup new Chat Template")
#     tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
# else:
#     print("Use predefined chat template:")
#     print(tokenizer.chat_template)
def formatting_prompts_func_nopacking(example):
    output_texts = []
    for i in range(len(example["prompt"])):
        text = f"### Question: {example['prompt'][i]}\n\n### Answer: {example['completion'][i]}\n\n"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func(example):
    return f"### Question: {example['prompt']}\n\n### Answer: {example['completion']}\n\n"

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,  # .map(lambda x: {'text': tokenizer.apply_chat_template(x, tokenize=False)}),
    packing=True,
    # formatting_func=formatting_prompts_func_nopacking,
    formatting_func=formatting_prompts_func,
    max_seq_length=max_position_embeddings,
)

trainer.train()
