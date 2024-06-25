import torch
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from load_data import load_eval_data
from utils import get_parse, setup_seed, setLogger

from metric.metric import reward

import re
import pandas as pd
import logging

args = get_parse()
args.eval_type = 'drd2_qed_sa'

setLogger("ppo_train")
logger = logging.getLogger()

config = PPOConfig(
    model_name="model_weights/checkpoints/gpt2_ft/checkpoint-1360000",
    learning_rate=1e-5,
    batch_size=512,
    mini_batch_size=32,
    gradient_accumulation_steps=16
    # log_with="wandb",
)

# model = GPT2LMHeadModel.from_pretrained(config.model_name)
# ref_model = GPT2LMHeadModel.from_pretrained(config.model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = create_reference_model(model)
tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
# reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")


fg_model_path = './model_weights/smiles2iupac'
fg_tokenizer = GPT2Tokenizer.from_pretrained(fg_model_path) #gpt2-medium
fg_configuration = GPT2Config.from_pretrained(fg_model_path, output_hidden_states=False)
fg_model = GPT2LMHeadModel.from_pretrained(fg_model_path, config=fg_model_path)
fg_model.to(args.device)
fg_model.eval()


# dataset = load_eval_data(args)

def build_dataset(config):
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_eval_data(args)

    # ds = ds.rename_columns({"text": "review"})
    # ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # sample["input_ids"] = tokenizer.encode(f"### Human: {sample['desc']} Give me the possible SMILES. ")
        sample["input_ids"] = tokenizer.encode(f"### Human: {sample['desc']} \n### Assistant:")
        sample["query"] = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def get_task_info(eval_type):
    task_list = ['IsValid','FG']
    target_list = ['Valid','IN']

    target_tasks = ['drd2','gsk3']
    admet_tasks = ['bbb','hia']

    task = 'normal'
    for t in target_tasks:
        if t in eval_type:
            task = 'target'
            task_list.append(t.upper())
            target_list.append('CAN')

    for t in admet_tasks:
        if t in eval_type:
            task = 'admet'
            task_list.append(t.upper())
            target_list.append('CAN')

    if '_qed' in eval_type:
        task_list.append('QED')
        target_list.append('HIGH')

    if '_sa' in eval_type:
        task_list.append('SAs')
        target_list.append('HIGH')

    if 'logp' in args.eval_type:
        task_list.append('LogP')
        condition = int(args.eval_type.split('logp_')[-1])
        target_list.append(condition)

    return task, task_list, target_list

dataset = build_dataset(config)

ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collator
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    # "top_k": 20,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_length":256
}

rewarded_smiles = []
saving_flag = 1
iterration = 1
early_stop = 200
while len(rewarded_smiles) < 100000:
    s_rewarded_len = len(rewarded_smiles)
    logger.info(f"Start with {iterration}th iteration, current rewared smiles number is: {s_rewarded_len}")

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        #### Get response from gpt2
        response_tensors = []
        for query in query_tensors:
            # gen_len = output_length_sampler()
            # generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze())

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        smiles_list = [s.split('Assistant:')[-1].split('<|endoftext|>')[0].strip() for s in texts]
        # smiles_list = [s.split('Assistant:')[-1].strip() for s in texts]

        # print(smiles_list)

        data_for_rewards = []
        for i in range(len(batch["response"])):
            prompt = batch['query'][i]
            response = batch['response'][i]
            smiles = response.split('Assistant:')[-1].split('<|endoftext|>')[0].strip()
            fgs = []
            encodings_dict = fg_tokenizer('<|startoftext|>' + '<|startofsmiles|>' + smiles + '<|startofiupac|>')
            input_ids = torch.tensor(encodings_dict['input_ids']).unsqueeze(0).to(args.device)
            iupac_outputs = fg_model.generate(
                                    input_ids,
                                    # bos_token_id=random.randint(1,30000),
                                    # do_sample=True,   
                                    # top_k=5, 
                                    num_beams = 5,
                                    max_length = 512,
                                    # top_p=0.95, 
                                    num_return_sequences=5,
                                    pad_token_id = fg_tokenizer.pad_token_id,
                                    return_dict_in_generate=True
                                )
            for seq in iupac_outputs.sequences:
                output = fg_tokenizer.decode(seq, skip_special_tokens=True)
                iupac = output.split("<|startofiupac|>")[-1].strip()
                fg_list = re.split("[\s\[\],\(\)-.;]",iupac)
                filtered_fg_list = [item for item in fg_list if len(item)>1 and item[0].isnumeric() is False]
                fgs.extend(filtered_fg_list)

            data_for_rewards.append([prompt,smiles,fgs])

        # print(data_for_rewards)
        logger.info("Number of generated smiles: ", len(data_for_rewards))

        data_for_rewards_df = pd.DataFrame(data=data_for_rewards, columns=['prompt','smiles','fgs'])
        task, task_list, target_list = get_task_info(args.eval_type)
        rewards = reward(data_for_rewards_df,task, task_list, target_list, rewarded_smiles)

        ################################## print ######################################
        tmp_rewarded_smiles = []
        for i in range(len(rewards)):
            if rewards[i]:
                tmp_rewarded_smiles.append(data_for_rewards[i][1])
        logger.info("Number of rewarded smiles: ", len(tmp_rewarded_smiles))
        rewarded_smiles.extend(tmp_rewarded_smiles)

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    # ppo_trainer.save_model("model_weights/gpt2_ppo")

    e_rewarded_len = len(rewarded_smiles)
    logger.info(f"End with {iterration}th iteration, current rewared smiles number is: {e_rewarded_len}")
    logger.info("\n")

    if e_rewarded_len - s_rewarded_len == 0:
        early_stop = early_stop - 1
    else:
        early_stop = 50

    if early_stop <= 0:
        model.save_pretrained(f"model_weights/checkpoints/gpt2_ppo/checkpoint-{e_rewarded_len}")
        tokenizer.save_pretrained(f"model_weights/checkpoints/gpt2_ppo/checkpoint-{e_rewarded_len}")
        break

    if (e_rewarded_len/1000)%10 > saving_flag:
        model.save_pretrained(f"model_weights/checkpoints/gpt2_ppo/checkpoint-{e_rewarded_len}")
        tokenizer.save_pretrained(f"model_weights/checkpoints/gpt2_ppo/checkpoint-{e_rewarded_len}")
        saving_flag = saving_flag + 1


