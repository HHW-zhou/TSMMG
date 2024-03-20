from rdkit import Chem
from rdkit.Chem import QED
import sys
sys.path.append('../')

import os
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem import MACCSkeys
import pandas as pd

import random

def sr_target_qed_sa(fname):
    f =  open(fname,'r')
    lines = f.readlines()

    outputs = []
    predicts = []
    truth = []
    desc_list = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

            desc = tmp[0].strip().split(': ')[-1].strip()
            desc_list.append(desc)

            # break
        elif 'Reference' in line:
            tmp = line.split('Reference smiles: ')[1].strip()
            truth.append(tmp)
        else:
            pass

    nums = len(predicts)
    # print(nums)

    sn = 0
    for i in range(nums):
        smile = predicts[i]
        desc = desc_list[i]

        try:
            m=Chem.MolFromSmiles(smile)
        except Exception as e:              # 不能解析的话跳过
            continue

        if m is None:
            continue

        qed=QED.qed(m)
        sas = sascorer.calculateScore(m)
        if qed < 0.6 or sas < 0.67:
            continue

        outputs.append([0,0,desc,smile,qed,sas])

    return outputs

def admet(source_data,fname,outname,mode='p'):
    df = pd.read_csv(fname, sep=' ')
    data = df.values

    len_of_source_data = len(source_data)
    len_of_data = len(data)

    print("len_of_source_data===> ", len_of_source_data)
    print("len_of_data===> ", len_of_data)
    assert len_of_source_data == len_of_data

    positives = []
    negtives = []

    for i in range(len_of_data):
        desc = source_data[i][2]
        smiles = source_data[i][3]
        qed = source_data[i][4]
        sas = source_data[i][5]
        score = data[i][1]

        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)

        if score > 0.6:
            positives.append([desc,smiles,round(score,2),round(qed,2),round(sas,2)])
        else:
            negtives.append([desc,smiles,round(score,2),round(qed,2),round(sas,2)])

    if mode == 'p':
        # print(positives)
        print("Success number (positives): ", len(positives))
        outputs = positives
    elif mode == 'n':
        # print(negtives)
        print("Success number (negtives): ", len(negtives))
        outputs = negtives

    outputs = pd.DataFrame(data=outputs, columns=['desc','smiles', 'score', 'qed','sa'])
    outputs.to_csv(f'../log/admet_{outname}.csv', index=False)
    print(f"保存 .../log/admet_{outname}.csv 成功，共 {len(outputs)} 条数据。")

source_file = '../log/eval_gtp2_292772_f_hia_qed_sa_150.log'
source_data = sr_target_qed_sa(source_file)
admet(source_data, '../log/hia_qed_sa.txt', 'hia_qed_sa')