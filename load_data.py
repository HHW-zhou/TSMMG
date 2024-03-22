import pandas as pd
import random
from datasets import DatasetDict
import torch
from torch.utils.data import Dataset

import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

class GPT2Dataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.smiles = []

        for data in data_list:
            desc = data[2]
            smiles = data[3]

            encodings_dict = tokenizer('<|startoftext|>'+ desc + '<|startofsmiles|>' + smiles + '<|endoftext|>', 
                                                        truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.smiles.append(smiles)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.smiles[idx]

class GPT2Dataset_test(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.smiles = []

        for data in data_list:
            desc = data[2]
            smiles = data[3]

            encodings_dict = tokenizer('<|startoftext|>'+ desc + '<|startofsmiles|>', truncation=True, max_length=max_length, padding="max_length")
            # encodings_dict = tokenizer('<|startoftext|>'+ desc + '<|startofsmiles|>')

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.smiles.append(smiles)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.smiles[idx]

class GPT2Dataset_eval(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.smiles = []

        for data in data_list:
            desc = data[2]
            smiles = data[3]

            # encodings_dict = tokenizer('<|startoftext|>'+ desc + '<|startofsmiles|>', truncation=True, max_length=max_length, padding="max_length")
            encodings_dict = tokenizer('<|startoftext|>'+ desc + '<|startofsmiles|>')

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.smiles.append(smiles)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.smiles[idx]

def get_kinase(args, tokenizer):
    train_type = args.train_type

    fname = f'./data/train/{train_type}.csv'
    df = pd.read_csv(fname)

    train_data_list = df.values

    data_size = len(train_data_list)
    
    train_data = GPT2Dataset(train_data_list, tokenizer)

    return train_data, data_size

def get_chunks():
    chunks = []

    task_list = ['1','2','3','4','5','target','admet','BTK','3CL','FGFR4','KPCD3','EP2','EP4']

    for task in task_list:
        fname = f'./data/train/{task}.csv'
        chunk = pd.read_csv(fname, iterator=True)
        chunks.append(chunk)

    return chunks

def load_data(args, tokenizer, chunk):
    fetch_size = args.batch_size * args.gpu_num * 100 if args.sample_size > 50000 else args.sample_size
    # fetch_size = args.batch_size * args.gpu_num

    try:
        data_df = chunk.get_chunk(fetch_size)
        fetched_size = len(data_df)
        train_data_list = data_df.values
    except Exception as e:
        return None, 0

    train_data = GPT2Dataset(train_data_list, tokenizer)

    return train_data, fetched_size

def load_test_data(args, tokenizer):
    eval_type = args.eval_type
    voc_num = args.voc_num

    fname = './data/eval/n5.csv'
    data_df = pd.read_csv(fname)
    data_list = data_df.values

    test_data_list = data_list[-1000:]

    test_data = GPT2Dataset_test(test_data_list, tokenizer)

    return test_data

def load_eval_data(args, tokenizer):
    eval_type = args.eval_type
    voc_num = args.voc_num

    fname = './data/eval/n5.csv'
    data_df = pd.read_csv(fname)
    data_list = data_df.values

    if eval_type == 'n':    #normal
        test_data_list = data_list[:1000]
    elif eval_type == 's':  #shuffle
        test_data_list = generate_desc_shuffle(args, data_list[:1000])
    elif eval_type == 'd':   #drop
        test_data_list = generate_desc_drop(args, data_list[:1000])
    elif eval_type in ['dm','dl','ds']:   
        test_data_list = generate_desc_drop_ls(args, data_list[:1000])
    elif eval_type in ['EP2','EP4','EPHX2nEP4','EPHX2','EP2nEP4']:   
        test_data_list = generate_desc_ep(args, data_list[:1000])
    else:
        test_data_list = generate_eval_targets(args)

    test_data = GPT2Dataset_eval(test_data_list, tokenizer)

    return test_data

def generate_desc_shuffle(args, list_data):
    outputs = []

    t1 = "The molecule contains {};"
    t2 = " its logP is {};"
    t3 = " the Synthetic Accessibility score (SAscore) of it is {}."

    for item in list_data:
        cid = item[0]
        iupac = item[1]
        smiles = item[3]

        try:
            m=Chem.MolFromSmiles(smiles)
        except Exception as e:              # 不能解析的话跳过
            continue

        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score

        qed=QED.properties(m)               # QED analysis
        MW = qed.MW
        ALOGP = qed.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic
        HBA = qed.HBA
        HBD = qed.HBD
        PSA = qed.PSA
        ROTB = qed.ROTB
        AROM = qed.AROM
        ALERTS = qed.ALERTS

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        # 打乱功能团顺序
        random.shuffle(filtered_word_list)
        #####################

        components = ''
        for word in filtered_word_list:
            components = components + word + ', '
        components = components[:-2]
        
        desc = t1.format(components)        #功能团
        desc = desc + t2.format(round(ALOGP,2))         #logp
        desc = desc + t3.format(round(sas,2))      #sas

        outputs.append([cid,iupac,desc,smiles])
        
    return outputs

def generate_desc_drop(args, list_data):
    drop_ratio = args.drop_ratio
    outputs = []

    t1 = "The molecule contains {};"
    t2 = " its logP is {};"
    t3 = " the Synthetic Accessibility score (SAscore) of it is {}."

    for item in list_data:
        cid = item[0]
        iupac = item[1]
        smiles = item[3]

        try:
            m=Chem.MolFromSmiles(smiles)
        except Exception as e:              # 不能解析的话跳过
            continue

        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score

        qed=QED.properties(m)               # QED analysis
        MW = qed.MW
        ALOGP = qed.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic
        HBA = qed.HBA
        HBD = qed.HBD
        PSA = qed.PSA
        ROTB = qed.ROTB
        AROM = qed.AROM
        ALERTS = qed.ALERTS

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        tmp_word_list = filtered_word_list.copy()
        
        ###### drop
        word_num = len(tmp_word_list)
        drop_num = int(word_num * drop_ratio)
        drop_idx = random.sample([i for i in range(word_num)], drop_num)
        drop_word_list = [tmp_word_list[idx] for idx in drop_idx]
        for dword in drop_word_list:
            tmp_word_list.remove(dword)
        ######

        components = ''
        for word in tmp_word_list:
            components = components + word + ', '
        components = components[:-2]
        
        desc = t1.format(components)        #功能团
        desc = desc + t2.format(round(ALOGP,2))         #logp
        desc = desc + t3.format(round(sas,2))      #sas

        outputs.append([cid,iupac,desc,smiles])
        
    return outputs

def generate_desc_drop_ls(args, list_data):
    eval_type = args.eval_type
    drop_ratio = args.drop_ratio
    try:
        df = pd.read_csv(f'./data/eval/eval_{eval_type}_{drop_ratio}.csv')
        outputs = df.values
    except Exception as e:
        outputs = None

    if outputs is None:
        dm_list = []
        dl_list = []
        ds_list = []

        t1 = "The molecule contains {}."
        t2 = " Its logP is {}."
        t3 = " It has {} synthetic accessibility."

        for item in list_data:
            cid = item[0]
            iupac = item[1]
            smiles = item[3]

            try:
                m=Chem.MolFromSmiles(smiles)
            except Exception as e:              # 不能解析的话跳过
                continue

            sas = sascorer.calculateScore(m)       # Synthetic Accessibility score

            qed=QED.properties(m)               # QED analysis
            ALOGP = qed.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic

            word_list = re.split("[\s\[\],\(\)-.;]",iupac)
            filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
            if len(filtered_word_list) == 0:
                continue

            tmp_word_list = filtered_word_list.copy()
            
            ###### drop
            word_num = len(tmp_word_list)
            drop_num = int(word_num * drop_ratio)
            drop_idx = random.sample([i for i in range(word_num)], drop_num)
            drop_word_list = [tmp_word_list[idx] for idx in drop_idx]
            for dword in drop_word_list:
                tmp_word_list.remove(dword)
            ######

            components = ''
            for word in tmp_word_list:
                components = components + word + ', '
            components = components[:-2]
            
            # 都不要
            desc1 = t1.format(components)    #功能团
            dm_list.append([cid,iupac,desc1,smiles])

            # logp
            ALOGP = -1
            desc2 = t1.format(components)    #功能团
            desc2 = desc2 + t2.format(round(ALOGP,2))         #logp
            dl_list.append([cid,iupac,desc2,smiles])

            # sas
            sas = 'good'
            desc3 = t1.format(components)    #功能团
            desc3 = desc3 + t3.format(sas)          #sas
            ds_list.append([cid,iupac,desc3,smiles])

            dm_df = pd.DataFrame(data=dm_list, columns=['cid', 'iupac', 'desc', 'smiles'])
            dm_df.to_csv(f'./data/eval/eval_dm_{drop_ratio}.csv', index=False)

            dl_df = pd.DataFrame(data=dl_list, columns=['cid', 'iupac', 'desc', 'smiles'])
            dl_df.to_csv(f'./data/eval/eval_dl_{drop_ratio}.csv', index=False)

            ds_df = pd.DataFrame(data=ds_list, columns=['cid', 'iupac', 'desc', 'smiles'])
            ds_df.to_csv(f'./data/eval/eval_ds_{drop_ratio}.csv', index=False)

            if eval_type == 'dm':
                outputs = dm_list
            elif eval_type == 'dl':
                outputs = dl_list
            elif eval_type == 'ds':
                outputs = ds_list

    return outputs

def generate_desc_ep(args, list_data):
    eval_type = args.eval_type
    drop_ratio = 0.9

    outputs = []

    t1 = "The molecule contains {}."
    t2 = " It can bind to Prostanoid {} receptor."

    for item in list_data:
        cid = item[0]
        iupac = item[1]
        smiles = item[3]

        try:
            m=Chem.MolFromSmiles(smiles)
        except Exception as e:              # 不能解析的话跳过
            continue

        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score

        qed=QED.properties(m)               # QED analysis
        ALOGP = qed.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        tmp_word_list = filtered_word_list.copy()
        
        ###### drop
        word_num = len(tmp_word_list)
        drop_num = int(word_num * drop_ratio)
        drop_idx = random.sample([i for i in range(word_num)], drop_num)
        drop_word_list = [tmp_word_list[idx] for idx in drop_idx]
        for dword in drop_word_list:
            tmp_word_list.remove(dword)
        ######

        components = ''
        for word in tmp_word_list:
            components = components + word + ', '
        components = components[:-2]
        
        # 都不要
        desc2 = t1.format(components)    #功能团
        desc2 = desc2 + t2.format(eval_type)         #logp

        if eval_type == 'EPHX2nEP4':
            # desc2 = t1.format(components) + " It exhibits a high QED score, good synthetic accessibility, and binds to both Prostanoid EPHX2 and EP4 receptors."
            desc2 = "The molecule exhibits a high QED score, good synthetic accessibility, and binds to both Prostanoid EPHX2 and EP4 receptors."
        elif eval_type == 'EP2nEP4':
            # desc2 = t1.format(components) + " It exhibits a high QED score, good synthetic accessibility, and binds to both Prostanoid EP2 and EP4 receptors."
            desc2 = "The molecule exhibits a high QED score, good synthetic accessibility, and binds to both Prostanoid EP2 and EP4 receptors."

        outputs.append([cid,iupac,desc2,smiles])

    return outputs

def generate_eval_targets(args):
    eval_type = args.eval_type
    fname = f'./data/eval/eval_{eval_type}.csv'
    df = pd.read_csv(fname)
    outputs = df.values
    return outputs

def get_ep_train_data(tokenizer):
    ep2_file = './data/train/EP2.csv'
    df_ep2 = pd.read_csv(ep2_file)


    ephx2_file = './data/train/EPHX2.csv'
    df_ephx2 = pd.read_csv(ephx2_file)

    ep4_file = './data/train/EP4.csv'
    df_ep4 = pd.read_csv(ep4_file)


    df = pd.concat([df_ep2, df_ephx2, df_ep4])
    train_data_list = df.values

    data_size = len(train_data_list)
    
    train_data = GPT2Dataset(train_data_list, tokenizer)

    return train_data, data_size

if __name__ == '__main__':
    # get_targets_and_admet(None)
    pass