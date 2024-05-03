
import torch
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference



alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


inputs = tokenizer(
[
    alpaca_prompt.format(
        "Classify the given article as either positive or negative or mix sentiment.", # instruction
        "mujhe aap pasand nahi ho", # input
        "", # output - leave this blank for generation!
    )
]*1, return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
decodes = tokenizer.batch_decode(outputs)
for decode in decodes:
    print(decode.split("### Response:")[-1].replace("</s>",""))
    
# Create list of files
test_files = [
    "/home/gaurishthakkar/projects/data-augmentation-moomin-23/test/bangla_test.csv",
    "/home/gaurishthakkar/projects/data-augmentation-moomin-23/test/combine_test.csv",
    "/home/gaurishthakkar/projects/data-augmentation-moomin-23/test/hin_test.csv",
    "/home/gaurishthakkar/projects/data-augmentation-moomin-23/test/mag_test.csv",
    "/home/gaurishthakkar/projects/data-augmentation-moomin-23/test/maithili_test.csv"
]

for test_file in test_files:
    file_name = test_file.split("/")[-1]
    df_test = pd.read_csv(test_file)
    
    ids = []
    predictions = []
    
    for i, row in df_test.iterrows():
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Classify the given article as either positive or negative or mix sentiment.", # instruction
                row["sentence"], # input
                "", # output - leave this blank for generation!
            )
        ]*1, return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
        decodes = tokenizer.batch_decode(outputs)
        for decode in decodes:
            prediction = decode.split("### Response:")[-1].replace("</s>","").strip()
            print(row["ids"],prediction)
            ids.append(row["ids"])
            predictions.append(prediction)
    pd.DataFrame({"ids":ids, "sentiment":predictions}).to_csv("submissions/"+file_name,index=None)
    