from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from transformers import EarlyStoppingCallback

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/mistral-7b-bnb-4bit",
#     "unsloth/llama-2-7b-bnb-4bit",
#     "unsloth/llama-2-13b-bnb-4bit",
#     "unsloth/codellama-34b-bnb-4bit",
#     "unsloth/tinyllama-bnb-4bit",
# ]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",  # "unsloth/llama-2-7b-bnb-4bit", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

from trl import SFTTrainer
from transformers import TrainingArguments

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

train_languages = "hindi" # "additional" #"magahi" #"hindi"  # ['all',"hindi", "magahi","bangla"]
use_additional = True #"additional"

df_train = pd.read_csv("data/train_all.csv")
df_test = pd.read_csv("data/test_all.csv")

if train_languages != "all":
    print("currently training on", train_languages)
    df_train = df_train[df_train.language == train_languages]
    df_test = df_test[df_test == train_languages]
    
if use_additional:
    df_train_additional = pd.read_csv("data/additional_data_train.csv")
    df_test_additional = pd.read_csv("data/additional_data_test.csv")
    
    df_train_additional["language"]="hindi"
    df_test_additional["language"]="hindi"
    
    df_train = pd.concat([df_train,df_train_additional])
    df_test = pd.concat([df_test,df_test_additional])

dataset = DatasetDict(
    {
        "train": Dataset.from_pandas(df_train, split="train"),
        "test": Dataset.from_pandas(df_test, split="test"),
    }
)

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        report_to="tensorboard",
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=100,
        optim="adamw_8bit",
        eval_steps=100,
        num_train_epochs=20,
        save_strategy="steps",
        evaluation_strategy="steps",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        do_train=True,
        do_eval=True,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

import torch

torch.cuda.empty_cache()

trainer_stats = trainer.train()
print(trainer_stats)

trainer.save_model()
# tokenizer.save_pretrained(training_args.output_dir)
trainer.log_metrics("train", trainer_stats.metrics)
trainer.save_metrics("train", trainer_stats.metrics)
trainer.save_state()


metrics = trainer.evaluate()
# metrics["seed"] = SEED  # training_args.seed
# metrics["train_size"] = len(dataset_collection["train"])
# metrics["valid_size"] = len(dataset_collection["test"])
# metrics["test_size"] = len(dataset_collection["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Classify the given article as either positive or negative or mix sentiment.",  # instruction
            "mujhe aap pasand nahi ho",  # input
            "",  # output - leave this blank for generation!
        )
    ]
    * 1,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
tokenizer.batch_decode(outputs)
model.save_pretrained("lora_model")  # Local saving
