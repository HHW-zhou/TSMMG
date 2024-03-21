import os
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from utils import get_parse, setup_seed, analysis_mol, metric_MOSS, write_metrics, visualize_by_html
from load_data import load_eval_data

from tqdm import tqdm
import shutil
from weasyprint import HTML

args = get_parse()
device = args.device
setup_seed(args.seed)

#########################################logger################################################
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

log_name = f'./log/eval_TSMMG.log'


logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)
###############################################################################################

model_path = args.eval_model_path
config_path = args.eval_model_path
tokenizer_path = args.eval_model_path

# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path) #gpt2-medium

test_data = load_eval_data(args, tokenizer)

# load model
configuration = GPT2Config.from_pretrained(config_path, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)

model.to(device)

def format_time(elapsed):
    return str(timedelta(seconds=int(round((elapsed)))))

# ========================================
#               Validation
# ========================================

logger.info("\n")
logger.info("Running Validation...")

t0 = time.time()

model.eval()

# Evaluate data for one epoch
outputs = []
mid = 1
for input_data in tqdm(test_data):
    
    input_ids = input_data[0].unsqueeze(0).to(device)
    ref_smiles = input_data[2]
    
    sample_outputs = model.generate(
                        input_ids,
                        do_sample=True,   
                        top_k=args.top_k, 
                        max_length = 512,
                        top_p=args.top_p, 
                        num_return_sequences = args.return_num,
                        pad_token_id = tokenizer.eos_token_id
                    )


    for j, sample_output in enumerate(sample_outputs):
        logger.info("{}: {}".format(j, tokenizer.decode(sample_output, skip_special_tokens=True)))
        logger.info(f"Reference smiles: {ref_smiles}")
        logger.info("\n")

        out_str = tokenizer.decode(sample_output, skip_special_tokens=True)
        smiles = out_str.split('<|startofsmiles|>')[-1].strip()

        # IsValid, QED, LogP, SAs = analysis_mol(smiles)

        # outputs.append([mid, smiles, IsValid, QED, LogP, SAs, ref_smiles])

        IsValid, QED, LogP, SAs = analysis_mol(smiles)

        outputs.append([mid, smiles, ref_smiles])

        mid = mid + 1
        

# 输出目录
# root_dir = f'./outputs/{args.eval_type}_' + datetime.now().strftime('%Y%m%d%H%M%S')
# fig_dir = f'{root_dir}/fig'
# html_dir = f'{root_dir}/html'
# if not os.path.exists(root_dir):
#     os.makedirs(root_dir)

# if not os.path.exists(fig_dir):
#     os.makedirs(fig_dir)

# if not os.path.exists(html_dir):
#     os.makedirs(html_dir)


# valid_ratio, unique_ratio = metric_MOSS(outputs)

# 输入生成的分子
# outputs_pd = pd.DataFrame(data=outputs, columns=['mid', 'smiles', 'IsValid', 'QED', 'LogP','SAs','ref_smiles'])
# outputs_pd.to_csv(f'{root_dir}/output.csv', index=False)

# outputs_fname = f'./outputs/{args.eval_type}_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
outputs_fname = f'./outputs/outputs_{args.eval_type}.csv'
# outputs_fname = f'{root_dir}/output.csv'

outputs_pd = pd.DataFrame(data=outputs, columns=['mid', 'smiles', 'ref_smiles'])
outputs_pd.to_csv(outputs_fname, index=False)


# 输入相关参数和度量数据
# info_list = [
#      ['Training sample size', args.sample_size],
#      ['Training epochs', args.epochs],
#      ['Eval task', args.eval_type],
#      ['Top k', args.top_k],
#      ['Top p', args.top_p],
#      ['Generative number', args.return_num],
#      ['Valid', valid_ratio],
#      ['Unique', unique_ratio]
# ]
# write_metrics(root_dir, info_list)

# 将HTML转换成PDF
# html_file = visualize_by_html(outputs, root_dir)
# pdf_file = f'{root_dir}/output.pdf'
# HTML(html_file).write_pdf(pdf_file)

# 删除图和pdf，只保留pdf
# if os.path.exists(fig_dir):
#     shutil.rmtree(fig_dir)

# if os.path.exists(html_dir):
#     shutil.rmtree(html_dir)

# os.system(f'zip -q -r {root_dir}.zip {root_dir}')

# if os.path.exists(root_dir):
#     shutil.rmtree(root_dir)

validation_time = format_time(time.time() - t0)    
logger.info("  Validation took: {:}".format(validation_time))




