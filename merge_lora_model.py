import sys
import os
# print(os.path.abspath('../'))
sys.path.append(os.path.abspath('../'))

# 指定CUDA设备号
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用第一个GPU
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

from utils import get_parse, merge_lora_weights

args = get_parse()

# 模型名称
model_name_or_path = "model_weights/backbones/Llama-2-7b-chat-hf"
# model_name_or_path = "model_weights/backbones/lora256-320000-merged"

# 下载并加载分词器
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
tokenizer.add_special_tokens({"pad_token":"[PAD]"})
tokenizer.padding_side = 'right'                       # decoder only 一般pad在左边，encoder pad在右边

if args.bf16:
    print("loading model with bf16......")
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=args.device)
elif args.fp16:
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map=args.device)
else:
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map=args.device)


model.resize_token_embeddings(len(tokenizer))   #Resize the embeddings
model.config.pad_token_id = tokenizer.pad_token_id  #Configure the pad token in the model
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

peft_checkpoint = f"model_weights/checkpoints/{args.backbone}_{args.training_strategy}/checkpoint-{args.checkpoint}"
# merge_lora_weights(model, peft_checkpoint)
model_to_merge  = PeftModel.from_pretrained(model, peft_checkpoint, device_map=args.device)

merged_model = model_to_merge.merge_and_unload()

saving_path = f"model_weights/checkpoints/{args.backbone}_{args.training_strategy}/checkpoint-{args.checkpoint}-merged"
# model.save_pretrained(saving_path)
merged_model.save_pretrained(saving_path)
tokenizer.save_pretrained(saving_path)