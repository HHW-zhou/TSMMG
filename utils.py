import argparse
import torch
import numpy as np
import random
import collections

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

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
    #    0        1         2        3      4      5        6
    #  'mid', 'smiles', 'IsValid', 'QED', 'LogP','SAs','ref_smiles'
    len_of_data = len(data)

    valid = []

    for item in data:
        smiles = item[1]
        IsValid = item[2]

        if IsValid:
            valid.append(smiles)
    
    valid_num = len(valid)
    valid_ratio = round(valid_num/len_of_data,4)

    unique_num = len(set(valid))
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
    parser.add_argument("--backbone", type=str, default='gpt2', choices=['biobert','bert','gpt2'])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=675354)
    parser.add_argument("--gpu_num", type=int, default=2)

    # for evaluation
    parser.add_argument("--train_type", type=str, default='BTK')  # normal/drop/shuffle
    parser.add_argument("--eval_type", type=str, default='n')  # normal/drop/shuffle
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--voc_num", type=int, default=1)    

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--return_num", type=int, default=5)

    parser.add_argument("--eval_model_path", type=str, default='./model_save_675354_0')  # normal/drop/shuffle

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    return args