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

def sr_qed_logp(fname, logp=2):
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
        desc = desc_list[i]
        smile = predicts[i]

        try:
            m=Chem.MolFromSmiles(smile)
            pptis=QED.properties(m)               # QED analysis
            ALOGP = pptis.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic
        except Exception as e:              # 不能解析的话跳过
            continue

        if abs(ALOGP-logp) > 1:
            continue

        if m is None:
            continue

        qed=QED.qed(m)
        if qed < 0.6:
            continue

        outputs.append([0,0,desc,smile])

    return outputs


def admet(source_data,fname,outname,mode='p'):
    df = pd.read_csv(fname, sep=' ')
    data = df.values

    len_of_source_data = len(source_data)
    len_of_data = len(data)

    assert len_of_source_data == len_of_data

    positives = []
    negtives = []

    for i in range(len_of_data):
        desc = source_data[i][2]
        smiles = source_data[i][3]
        score = data[i][1]

        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)

        ppties=QED.properties(m)               # QED analysis
        ALOGP = ppties.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic

        if score > 0.6:
            positives.append([desc,smiles,round(qed,2),round(ALOGP,2),round(score,2)])
        else:
            negtives.append([desc,smiles,round(qed,2),round(ALOGP,2),round(score,2)])

    if mode == 'p':
        # print(positives)
        print("Success number (positives): ", len(positives))
        outputs = positives
    elif mode == 'n':
        # print(negtives)
        print("Success number (negtives): ", len(negtives))
        outputs = negtives

    outputs = pd.DataFrame(data=outputs, columns=['desc','smiles', 'qed','logp', 'score'])
    outputs.to_csv(f'../log/admet_{outname}.csv', index=False)
    print(f"保存 .../log/admet_{outname}.csv 成功，共 {len(outputs)} 条数据。")

    

# source_file = '../log/eval_gtp2_292772_f_hia_qed_sa_150.log'
# source_data = sr_qed_logp(source_file)
# admet(source_data, '../log/hia_qed_sa.txt', 'hia_qed_sa')


# source_file = '../log/eval_gtp2_1377731_bd_40.log'
# source_data = sr_qed_logp(source_file)
# admet(source_data, '../log/bd.txt', 'bd')

# source_file = '../log/eval_gtp2_1377731_bd_ft_40.log'
# source_data = sr_qed_logp(source_file)
# admet(source_data, '../log/bd_ft_bd.txt', 'bd_ft_bd')


# ###############################################

# source_file = '../log/eval_gtp2_1377731_ft_40.log'
# source_data = sr_qed_logp(source_file)
# admet(source_data, '../log/ft.txt', 'ft','n')

# source_file = '../log/eval_gtp2_1377731_bd_ft_40.log'
# source_data = sr_qed_logp(source_file)
# admet(source_data, '../log/bd_ft_ft.txt', 'bd_ft_ft','n')



def mix():
    f1 = '../log/admet_bd_ft_bd.csv'
    df1 = pd.read_csv(f1)
    data1 = df1.values

    f2 = '../log/admet_bd_ft_ft.csv'
    df2 = pd.read_csv(f2)
    data2 = df2.values

    outputs = []
    for i in range(len(data1)):
        desc = data1[i][0]
        smiles1 = data1[i][1]
        qed = data1[i][2]
        logp = data1[i][3]
        score1 = data1[i][4]

        for j in range(len(data2)):
            smiles2 = data2[j][1]
            score2 = data2[j][4]

            if smiles1 == smiles2:
                outputs.append([desc,smiles1,qed,logp,score1,score2])


    outputs = pd.DataFrame(data=outputs, columns=['desc','smiles','qed','logp','bd','ft'])
    outputs.to_csv(f'../log/result_bd_ft.csv', index=False)
    print(f"保存 .../log/result_bd_ft.csv 成功，共 {len(outputs)} 条数据。")

# mix()


# def target_qed_sa(source_data,fname,outname,mode='p'):
#     df = pd.read_csv(fname, sep=' ')
#     data = df.values

#     len_of_source_data = len(source_data)
#     len_of_data = len(data)

#     assert len_of_source_data == len_of_data

#     positives = []
#     negtives = []

#     for i in range(len_of_data):
#         desc = source_data[i][2]
#         smiles = source_data[i][3]
#         score = data[i][1]

#         m=Chem.MolFromSmiles(smiles)
#         qed=QED.qed(m)

#         ppties=QED.properties(m)               # QED analysis
#         ALOGP = ppties.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic

#         if score > 0.6:
#             positives.append([desc,smiles,round(qed,2),round(ALOGP,2),round(score,2)])
#         else:
#             negtives.append([desc,smiles,round(qed,2),round(ALOGP,2),round(score,2)])

#     if mode == 'p':
#         # print(positives)
#         print("Success number (positives): ", len(positives))
#         outputs = positives
#     elif mode == 'n':
#         # print(negtives)
#         print("Success number (negtives): ", len(negtives))
#         outputs = negtives

#     outputs = pd.DataFrame(data=outputs, columns=['desc','smiles', 'qed','logp', 'score'])
#     outputs.to_csv(f'../log/admet_{outname}.csv', index=False)
#     print(f"保存 .../log/admet_{outname}.csv 成功，共 {len(outputs)} 条数据。")

    

# source_file = '../log/eval_gtp2_292772_f_hia_qed_sa_150.log'
# source_data = sr_qed_logp(source_file)
# admet(source_data, '../log/hia_qed_sa.txt', 'hia_qed_sa')