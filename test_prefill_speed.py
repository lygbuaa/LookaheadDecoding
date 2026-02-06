from transformers import AutoTokenizer ##, AutoModelForCausalLM
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
)
import torch
import time 
import os 
if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 
    ## for TinyLlama-1.1B-Chat-v1.0 @cuda
    lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=1, POOL_FROM_PROMPT=True)
    ## for TinyLlama-1.1B-Chat-v1.0 @cpu
    # lade.config_lade(LEVEL=3, WINDOW_SIZE=5, GUESS_SET_SIZE=5, DEBUG=1, POOL_FROM_PROMPT=True)

# assert torch.cuda.is_available()
# torch_device = "cuda"
torch_device = "cpu"

SEQ_LEN = 1
LOOP = 4
TOTAL_DT = 0
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "/home/hugoliu/alaska/ws_llm/hf_data/TinyLlama-1.1B-Chat-v1.0"
# model_path = "/home/hugoliu/alaska/ws_llm/hf_data/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=torch_device)
model = LlamaForCausalLM.from_pretrained(
    model_path, 
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    offload_folder="offload",
    device_map=torch_device
)

model.tokenizer = tokenizer
model_config = model.config
pad_token_id = model_config.pad_token_id
bos_token_id = model_config.bos_token_id
eos_token_id = model_config.eos_token_id
print(f"model_path: {model_path}, pad_token_id: {pad_token_id}, bos_token_id: {bos_token_id}, eos_token_id: {eos_token_id}")

prompt = "How do you fine tune a large language model?"
input_text = (
    f"<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
)

model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

model_inputs["input_ids"] = torch.randint(low=8, high=1024, size=[1, SEQ_LEN], dtype=torch.int32).to(torch_device)
model_inputs["attention_mask"] = torch.ones(1, SEQ_LEN, dtype=torch.int32).to(torch_device)

for k, v in model_inputs.items():
    print(f"[model_inputs] {k}: {v.shape if isinstance(v, torch.Tensor) else v}")

#warm up
greedy_output = model.generate(**model_inputs, max_new_tokens=1)
#end warm up

for i in range(LOOP):
    torch.cuda.synchronize()
    t0g = time.time()
    greedy_output = model.generate(**model_inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    t1g = time.time()
    dt = t1g - t0g
    TOTAL_DT += dt
    print(f"[greedy][{i}] dt = {dt}")
    # print("Greedy output: ", tokenizer.decode(greedy_output[0], skip_special_tokens=False))

print(f"[greedy] avg dt of {LOOP} = {TOTAL_DT/LOOP}")
#python minimal.py #44 tokens/s
#LOAD_LADE=1 USE_LADE=1 python minimal.py #74 tokens/s, 1.6x throughput without changing output distribution!

