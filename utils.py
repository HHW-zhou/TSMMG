import argparse
import torch
import numpy as np
import random
import collections
import logging
import time

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def write_html(root_dir, cyclic_id, divs):
    prefix ='''
                <!DOCTYPE html>
                <html>
                    <head>
                    <meta charset="UTF-8">
                    <title></title>
                    <style>
                        .base{
                        width: 300px;
                        height: 300px;
                        display: inline-block;
                        word-break: break-all
                        }
                        .hit{
                        width: 300px;
                        height: 300px;
                        display: inline-block;
                        word-break: break-all;
                        color:green
                        }
                        .half_hit{
                        width: 300px;
                        height: 300px;
                        display: inline-block;
                        word-break: break-all;
                        color:orange
                        }
                        .no_hit{
                        width: 300px;
                        height: 300px;
                        display: inline-block;
                        word-break: break-all;
                        color:red
                        }
                    </style>
                    </head>
                    <body>
                    <div id="out">
            '''
    suffix ='''
                    </div>
                    </body>
                </html>
            '''


    contents = ''

    for div in divs:
        contents = contents + div

    html = prefix + contents + suffix
    
    html_file = f'{root_dir}/html/{cyclic_id}.html'

    with open(html_file,'w') as f:
        f.write(html)

    return html_file

def add_div(url, property_dict, bgcolor):
    if 'QED' in property_dict:
        qed = property_dict['QED']
        LogP = property_dict['LogP']
        SAs = property_dict['SAs']
        style = 'no_hit'
        if qed > 0.6 or SAs < 4:
            style = 'half_hit'

        if LogP > 1 and LogP < 5:
            style = 'half_hit'

        if qed > 0.6 and SAs < 4 and LogP >1 and LogP < 5:
            style = 'hit'
    else:
        style = 'base'

    properties = ""

    for key, value in property_dict.items():
        tmp = f'''{key}: {value} &nbsp'''
        properties = properties + tmp

    new_div = f'''
                    <div class="{style}">
                        <img src="{url}" height="300px" width="300px">
                        <p style="background-color:{bgcolor}">{properties}</p>
                    </div>

                '''
    return new_div

def get_div_properties(*args):
    property_dict = collections.OrderedDict()
    for arg in args:
        key = arg[0]
        value = arg[1]
        property_dict[key] = value
    return property_dict

def visualize_by_html(data, root_dir):
    #  0       1         2        3      4      5          6      
    # mid, 'smiles', 'IsValid', 'QED', 'LogP','SAs','ref_smiles'

    divs = []
    for item in data:
        mid = item[0]
        smiles = item[1]
        IsValid = item[2]
        qed = item[3]
        LogP = item[4]
        SAs = item[5]
        ref_smiles = item[6]

        if not IsValid:
            continue

        mol = Chem.MolFromSmiles(smiles)
        fname_m = f'{mid}.svg'
        fpath_m = f'{root_dir}/fig/{fname_m}'
        url_m = f'../fig/{fname_m}'

        # rdCoordGen.AddCoords(mol)
        Draw.MolToFile(mol, fpath_m)

        property_dict = get_div_properties(
            ['file name',fname_m],
            ['smiles',smiles],
            ['QED',qed],
            ['LogP',LogP],
            ['SAs',SAs],
            ['Ref',ref_smiles],
            )
    
        div = add_div(url_m, property_dict, 'white')
        divs.append(div)

    html_file = write_html(root_dir, 'output', divs)

    return html_file

def write_metrics(root_dir, info_list):
    fname = f'{root_dir}/metrics.txt'

    with open(fname, 'w') as f:
        for info in info_list:
            content = f'{info[0]}: {info[1]}\n'
            f.write(content)

def metric_MOSS(data):
    len_of_data = len(data)
    valid_smiles = [item['smiles'] for item in data if item['IsValid'] is True]
    
    valid_num = len(valid_smiles)
    valid_ratio = round(valid_num/len_of_data,4)

    unique_num = len(set(valid_smiles))
    unique_ratio = round(unique_num/valid_num,4)

    return valid_ratio, unique_ratio

def analysis_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False,False,False,False

    qed=QED.qed(mol)
    qed = round(qed,2)

    qed_=QED.properties(mol)
    LogP = qed_.ALOGP
    LogP = round(LogP,2)

    SAs = sascorer.calculateScore(mol)
    SAs = round(SAs,2)

    return True, qed, LogP, SAs

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_parse():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--backbone", type=str, default='gpt2', choices=['gpt2','llama-7b','gpt2_bck'])
    parser.add_argument("--model_name_or_path", type=str, default='model_weights/backbones/Llama-2-7b-chat-hf')
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=15000000)
    parser.add_argument("--gpu_num", type=int, default=2)

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="lvwerra/stack-exchange-paired")
    parser.add_argument("--subset", type=str, default="data/finetune")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--samples_to_update_gradient", type=int, default=512)      # 每多少个样本更新一次参数
    parser.add_argument("--save_freq", type=int, default=10000)                       # 每更新多少次参数保存一次

    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")          # constant/cosine
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_rank", type=int, default=128)

    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./model_weights/checkpoints")
    parser.add_argument("--training_strategy", type=str, default="lora_sft") # 'lora_sft','lora_ft','sft','ft','dpo','ppo'
    parser.add_argument("--quantify", type=str, default="8bit", choices=['4bit','8bit','no'])
    parser.add_argument("--report_to", type=str, default="wandb", choices=['wandb','none'])
    parser.add_argument("--checkpoint", type=str, default="final")

    # for evaluation
    parser.add_argument("--train_type", type=str, default='BTK')  # normal/drop/shuffle
    parser.add_argument("--eval_type", type=str, default='drd2_short')  # normal/drop/shuffle
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--voc_num", type=int, default=1)    
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--return_num", type=int, default=5)
    parser.add_argument("--eval_model_path", type=str, default='./model_save_675354_0')  # normal/drop/shuffle
    parser.add_argument("--T", type=float, default=1.0)

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    args.gpu_num = torch.cuda.device_count()
    args.gradient_accumulation_steps = int(args.samples_to_update_gradient / args.gpu_num / args.batch_size)
    args.output_dir = os.path.join(args.output_dir,f'{args.backbone}_{args.training_strategy}')

    return args

def setLogger(task=None):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件

    if task is not None:
        rq = task
    else:
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logfile = f'{log_dir}/{rq}.log'
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)


def merge_lora_weights(model, lora_path):
    state_dict = get_fp32_state_dict_from_zero_checkpoint(lora_path)

    proj_type_list = ['q_proj','k_proj','v_proj','o_proj']
    # proj_type_list = ['q_proj','v_proj']


    for name, param in model.named_parameters():
        if 'layers' in name:
            layer_number = name.split("model.layers.")[-1].split('.')[0]

            for proj_type in proj_type_list:
                if proj_type in name:
                    key_A = f'base_model.model.model.layers.{layer_number}.self_attn.{proj_type}.lora_A.default.weight'
                    key_B = f'base_model.model.model.layers.{layer_number}.self_attn.{proj_type}.lora_B.default.weight'

                    lora_A_weight = state_dict[key_A].to(param.data.device)
                    lora_B_weight = state_dict[key_B].to(param.data.device)

                    print(lora_A_weight)

                    delta_w = torch.mm(lora_B_weight,lora_A_weight)
                    param.data = param.data + delta_w

                    print(f"layer {layer_number}, {proj_type} merged.")