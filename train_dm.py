import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# print(os.path.abspath('../'))
sys.path.append(os.path.abspath('../'))

print(sys.path)

# 指定CUDA设备号
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第一个GPU

import torch
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from transformers import LlamaTokenizer, LlamaForCausalLM,  GPT2LMHeadModel,  GPT2Tokenizer

from accelerate import Accelerator

from load_data import load_dm_data
from utils import get_parse, setup_seed, setLogger
# from accelerate.utils import DummyOptim, DummyScheduler

def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


accelerator = Accelerator()

# print("===========> ", accelerator.device)
# print("===========> ", accelerator.device)

#logger
setLogger()
logger = logging.getLogger()

args = get_parse()
setup_seed(args.seed)

# 下载并加载分词器
if args.backbone == 'llama-7b':
    # 模型名称
    # model_name_or_path = "model_weights/backbones/Llama-2-7b-chat-hf"
    model_name_or_path = args.model_name_or_path
    # model_name_or_path = "model_weights/checkpoints/llama-7b_lora_ft_v0/checkpoint-320000-merged"
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    # smiles_token = "<|startofsmiles|>"                    # 防止该标志被分解成多个token
    # tokenizer.add_tokens([smiles_token])
    tokenizer.padding_side = 'right'                       # decoder only 一般pad在左边，encoder pad在右边

    if args.quantify == '4bit':
        logger.info("Load model with 4bit quantization ...... ")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
        )
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map={"": accelerator.device})
    elif args.quantify == '8bit':
        logger.info("Load model with 8bit quantization ...... ")
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
        )
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map={"": accelerator.device})
    else:
        logger.info("Load model without quantization ...... ")
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map={"": accelerator.device})

    model.resize_token_embeddings(len(tokenizer))   #Resize the embeddings
    model.config.pad_token_id = tokenizer.pad_token_id   #Configure the pad token in the model
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj","lm_head"],
        # target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

elif args.backbone == 'gpt2':
    # 模型名称
    model_name_or_path = f"model_weights/checkpoints/{args.backbone}_ft/checkpoint-{args.checkpoint}"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    # smiles_token = "<|startofsmiles|>"                    # 防止该标志被分解成多个token
    # tokenizer.add_tokens([smiles_token])
    tokenizer.padding_side = 'right'                       # decoder only 一般pad在左边，encoder pad在右边

    if args.quantify == '4bit':
        logger.info("Load model with 4bit quantization ...... ")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
        )
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map={"": accelerator.device})
    elif args.quantify == '8bit':
        logger.info("Load model with 8bit quantization ...... ")
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
        )
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, quantization_config=bnb_config, device_map={"": accelerator.device})
    else:
        logger.info("Load model without quantization ...... ")
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path, device_map={"": accelerator.device})

    
    model.resize_token_embeddings(len(tokenizer))   # #Resize the embeddings
    model.config.pad_token_id = tokenizer.pad_token_id  #Configure the pad token in the model
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


logger.info("Loading data ... ")
dataset = load_dm_data(args, tokenizer)
logger.info(f"Data loaded, data length: {len(dataset)}")

training_arguments = TrainingArguments(
        # device='cuda:0',
        do_eval=False,
        output_dir=args.output_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        log_level="debug",
        optim="paged_adamw_32bit",
        save_steps=args.gradient_accumulation_steps*args.save_freq,
        logging_steps=args.gradient_accumulation_steps*args.save_freq,          # logging 频率和参数更新频率保持一致
        eval_steps=args.gradient_accumulation_steps*args.save_freq,
        fp16=args.fp16,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        num_train_epochs=args.epochs,
        # max_steps=1, #remove this
        warmup_ratio=0.03,
        lr_scheduler_type=args.lr_scheduler_type,                   # constant/cosine
        report_to= args.report_to,
        # gradient_checkpointing_kwargs={'use_reentrant':False},
        gradient_checkpointing = False,   # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        ddp_find_unused_parameters = False, #When using distributed training, the value of the flag find_unused_parameters passed to DistributedDataParallel. Will default to False if gradient checkpointing is used, True otherwise.
        # fsdp = 'full_shard'
        # deepspeed='/mnt/ai4s_ceph_share/neogaryzhou/TSMMG/scripts/zero_stage3_config.json'
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['desc'])):
        # text = f"### Human: {example[i]['desc']} Give me the possible SMILES. ### Assistant: {example[i]['smiles']}"
        # text = f"### Human: {example['desc'][i]} Give me the possible SMILES. \n### Assistant: {example['smiles'][i]}"
        text = f"### Human: {example['desc'][i]} \n### Assistant: {example['smiles'][i]}"
        # print(text)
        output_texts.append(text)
    return output_texts

response_template = "\n### Assistant:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.seq_length,
        dataset_num_proc=args.num_workers,                          # The number of workers to use to tokenize the data. Only used when packing=False. Defaults to None.
        # formatting_func=formatting_prompts_func,
        data_collator=collator if  args.training_strategy in ['sft','lora_sft'] else None,
        peft_config=peft_config if args.training_strategy in ['lora_sft','lora_ft'] else None,
        packing=False,
)

print_trainable_parameters(trainer.model, logger)

logger.info("Training...")
# trainer.train(resume_from_checkpoint=True)
# trainer.train(resume_from_checkpoint="model_weights/checkpoints/gpt2_ft/checkpoint-340000")
trainer.train()

logger.info("Saving last checkpoint of the model")
trainer.model.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint-final"))

sys.exit()