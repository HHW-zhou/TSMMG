# 不再使用 <|startofsmiles|> 分割，使用 assistant

import sys
import os
import re

# print(os.path.abspath('../'))
sys.path.append(os.path.abspath('../'))

# 指定CUDA设备号
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用第一个GPU

import torch
import pandas as pd
from peft import PeftModel
from transformers import BitsAndBytesConfig, GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from load_data import load_eval_data
from utils import get_parse, setup_seed, analysis_mol, metric_MOSS, write_metrics, visualize_by_html, merge_lora_weights

args = get_parse()
setup_seed(args.seed)

if args.backbone == 'llama-7b':
    # 模型名称
    # model_name_or_path = "model_weights/backbones/Llama-2-7b-chat-hf"
    # model_name_or_path = args.model_name_or_path
    # model_name_or_path = "model_weights/checkpoints/llama-7b_lora_ft_v0/checkpoint-320000-merged"

    model_name_or_path = f"model_weights/checkpoints/{args.backbone}_{args.training_strategy}/checkpoint-{args.checkpoint}-merged"

    # 下载并加载分词器
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.padding_side = 'right'                       # decoder only 一般pad在左边，encoder pad在右边

    if args.quantify == '4bit':
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
        )
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map=args.device)
    elif args.quantify == '8bit':
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
        )
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map=args.device)
    else:
        if args.bf16:
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=args.device)
        elif args.fp16:
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map=args.device)
        else:
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map=args.device)

    
    model.resize_token_embeddings(len(tokenizer))   #Resize the embeddings
    model.config.pad_token_id = tokenizer.pad_token_id  #Configure the pad token in the model
    # model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

    # if 'lora' in args.training_strategy:
    #     peft_checkpoint = f"model_weights/checkpoints/{args.backbone}_{args.training_strategy}/checkpoint-{args.checkpoint}"
    #     merged_model = PeftModel.from_pretrained(model, peft_checkpoint, device_map=args.device)
    #     eval_model=merged_model

    #     # merge_lora_weights(model, peft_checkpoint)
    # else:
    #     eval_model = model

    eval_model = model

elif args.backbone == 'gpt2':
    model_name_or_path = f"model_weights/checkpoints/{args.backbone}_{args.training_strategy}/checkpoint-{args.checkpoint}"
    # model_name_or_path2 = f"model_weights/checkpoints/gpt2_sft/checkpoint-160000"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = 'right'                       # decoder only 一般pad在左边，encoder pad在右边

    if args.quantify == '4bit':
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
        )
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map=args.device)
    elif args.quantify == '8bit':
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
        )
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map=args.device)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, device_map=args.device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    eval_model = model

def generate(instruction):
    tmp_smiles_list = []

    if args.backbone in ['llama-7b'] or args.training_strategy in ['ft','sft']:
        # prompt = f"### Human: {instruction} Give me the possible SMILES. \n### Assistant:"
        prompt = f"### Human: {instruction} \n### Assistant:"
    else:
        # prompt = f"### Human: {instruction} Give me the possible SMILES. "
        prompt = f"### Human: {instruction}"

    inputs = tokenizer(prompt, return_tensors="pt")
    # print(inputs["input_ids"])
    # print(tokenizer.decode(inputs["input_ids"][0]))
    input_ids = inputs["input_ids"].to(args.device)
    
    # generation_output = merged_model.generate(
    #         do_sample=True,   
    #         input_ids=input_ids,
    #         generation_config=GenerationConfig(temperature=1.0, top_p=0.95, top_k=5),
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #         max_new_tokens=256,
    #         pad_token_id=tokenizer.pad_token_id,
    #         num_return_sequences = 5
    # )

    generation_output = eval_model.generate(
            input_ids,
            do_sample=True,   
            temperature=args.T,
            top_k=args.top_k, 
            max_length = 256,
            top_p=args.top_p, 
            num_return_sequences = args.return_num,
            pad_token_id = tokenizer.pad_token_id,
            return_dict_in_generate=True
    )

    for seq in generation_output.sequences:
        output = tokenizer.decode(seq, skip_special_tokens=True)
        print("\n", output)
        # print(output.split("### Assistant: "))
        smiles = output.split("### Assistant: ")[-1].strip()
        tmp_smiles_list.append(smiles)
        # print("ouput:", smiles)

    return tmp_smiles_list

outputs = []
eval_dataset = load_eval_data(args)
for eval_data in eval_dataset:
    tmp_smiles_list = generate(eval_data['desc'])

    for smiles in tmp_smiles_list:
        # IsValid, QED, LogP, SAs = analysis_mol(smiles)
        outputs.append({
            'prompt':eval_data['desc'],
            'smiles':smiles,
            # 'IsValid':IsValid,
            # 'QED':QED,
            # 'LogP':LogP,
            # 'SAs':SAs
        })


# df_outputs = pd.DataFrame(outputs)
# save_name = f'./outputs/output_{args.backbone}_{args.training_strategy}_{args.eval_type}.csv'
# df_outputs.to_csv(save_name, index=False)

# valid_ratio, unique_ratio = metric_MOSS(outputs)
# print(f"Valid: {valid_ratio}, Unique: {unique_ratio}")

del model
del eval_model

############################################ FG retrival ###############################################
fg_model_path = './model_weights/smiles2iupac'
fg_tokenizer = GPT2Tokenizer.from_pretrained(fg_model_path) #gpt2-medium
fg_configuration = GPT2Config.from_pretrained(fg_model_path, output_hidden_states=False)
fg_model = GPT2LMHeadModel.from_pretrained(fg_model_path, config=fg_model_path)
fg_model.to(args.device)
fg_model.eval()

for i in range(len(outputs)):
    smiles = outputs[i]['smiles']

    encodings_dict = fg_tokenizer('<|startoftext|>' + '<|startofsmiles|>' + smiles + '<|startofiupac|>')
    input_ids = torch.tensor(encodings_dict['input_ids']).unsqueeze(0).to(args.device)
    iupac_outputs = fg_model.generate(
                            input_ids,
                            # do_sample=True,   
                            # temperature=1,
                            # top_k=10, 
                            # top_p=args.top_p, 
                            num_beams = 5,
                            max_length = 256,
                            num_return_sequences=5,
                            pad_token_id = fg_tokenizer.pad_token_id,
                            return_dict_in_generate=True
                        )
    for seq in iupac_outputs.sequences:
        output = fg_tokenizer.decode(seq, skip_special_tokens=True)
        iupac = output.split("<|startofiupac|>")[-1].strip()
        fg_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_fg_list = [item for item in fg_list if len(item)>1 and item[0].isnumeric() is False]
        print(iupac,filtered_fg_list)

        if 'fgs' in outputs[i]:
            outputs[i]['fgs'].extend(filtered_fg_list)
        else:
            outputs[i]['fgs'] = filtered_fg_list


df_outputs = pd.DataFrame(outputs)
save_name = f'./outputs/output_{args.backbone}_{args.training_strategy}_{args.eval_type}.csv'
df_outputs.to_csv(save_name, index=False)

    