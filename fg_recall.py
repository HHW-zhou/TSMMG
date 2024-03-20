from STOUT import translate_forward, translate_reverse
from tqdm import tqdm
import time
from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd
import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    # pre-parsing args
    parser.add_argument("--sample_size", type=str, default='1000k')
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--eval_type", type=str, default='n')  # normal/drop/shuffle
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--voc_num", type=int, default=1)
    args = parser.parse_args()
    return args

def generate_iupac(fname):
    # out：
    #   truth：输入的文本对应的SMILES，不一定存在
    #   voc_list：输入文本的功能团串，由逗号隔开
    #   predicts：预测的SMILES
    #   predict_iupac_list：根据预测的SMILES生成的IUPAC NAME；若预测的SMILES非法，则为'None'

    # f =  open('./log/eval_gtp2_30k.log','r')
    f =  open(fname,'r')
    lines = f.readlines()
    # print(len(lines))

    predicts = []
    truth = []
    voc_list = []
    for line in lines:
        if 'startofsmiles' in line:
            tmp = line.split('<|startofsmiles|>')[1].strip()
            predicts.append(tmp)

            # 输入的功能团
            # vocs = line.replace(' ','').split('contains')[1].split(';')[0].split(',')
            vocs = line.replace(' ','').split('contains')[1].split(';')[0]
            voc_list.append(vocs)

            # break
        elif 'Reference' in line:
            tmp = line.split('Reference smiles: ')[1].strip()
            truth.append(tmp)
        else:
            pass
    
    nums = len(truth)
    # print(nums)

    predict_iupac_list = []
    outputs = []
    for i in tqdm(range(nums)):
        # print(predicts[i])
        SMILES = predicts[i]
        m = Chem.MolFromSmiles(SMILES)
        if m is not None:                                   #如果预测的SMILES有效，则计算IUPAC_NAME
            IUPAC_name = translate_forward(SMILES)
        else:
            IUPAC_name = 'Invalid'
        predict_iupac_list.append(IUPAC_name)

        outputs.append([truth[i],voc_list[i],predicts[i],IUPAC_name])

    # return truth, voc_list, predicts, predict_iupac_list
    return outputs

args = get_parse()

if args.eval_type in ['d','dm','dl','ds','drd2']:
    fname = f'./log/eval_gtp2_{args.sample_size}_{args.eval_type}_{args.epochs}_{args.drop_ratio}.log'
    sfname = f'./log/eval_gtp2_{args.sample_size}_{args.eval_type}_{args.epochs}_{args.drop_ratio}.csv'
else:
    fname = f'./log/eval_gtp2_{args.sample_size}_{args.eval_type}_{args.epochs}.log'
    sfname = f'./log/eval_gtp2_{args.sample_size}_{args.eval_type}_{args.epochs}.csv'

# def generate_iupac(fname):
outputs = generate_iupac(fname)

df = pd.DataFrame(data=outputs, columns=['truth','vocs','pred_SMILES','pred_IUPAC'])
df.to_csv(sfname, index=False)