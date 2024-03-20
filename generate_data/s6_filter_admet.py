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
import math


def maccs_from_mol(mol):
    maccs = [1]
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_bits = list(fp.GetOnBits())
    for fp in fp_bits:
        maccs.append(f'{fp}:1')
    return maccs

def sr_target(fname, fname2, target):
    f =  open(fname,'r')
    lines = f.readlines()

    predicts = []
    truth = []
    desc_list = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

            if 'INFO: 0: ' in line:
                split_token = 'INFO: 0: '
            elif 'INFO: 1: ' in line:
                split_token = 'INFO: 1: '
            elif 'INFO: 2: ' in line:
                split_token = 'INFO: 2: '
            elif 'INFO: 3: ' in line:
                split_token = 'INFO: 3: '
            elif 'INFO: 4: ' in line:
                split_token = 'INFO: 4: '
            else:
                pass

            desc = tmp[0].strip().split(split_token)[-1].strip()
            desc_list.append(desc)

            # break
        elif 'Reference' in line:
            tmp = line.split('Reference smiles: ')[1].strip()
            truth.append(tmp)
        else:
            pass

    nums = len(predicts)
    # print(nums)

    idx_list = []
    for i in range(nums):
        smile = predicts[i]

        try:
            m=Chem.MolFromSmiles(smile)
        except Exception as e:              # 不能解析的话跳过
            continue

        if m is None:
            continue

        # maccs = maccs_from_mol(m)
        # idx = math.floor(i/25)
        # idx = math.floor(i)
        idx = i
        idx_list.append(idx)

    ##############################################################
    df = pd.read_csv(fname2, sep=' ')
    score_list = df['1'].values
    
    len_of_score_list = len(score_list)
    len_of_idx_list = len(idx_list)

    assert len_of_score_list == len_of_idx_list

    top6 = []
    top4 = []
    top2 = []
    top0 = []
    for i in range(len_of_score_list):
        score = score_list[i]
        idx = idx_list[i]
        if score > 0.6:
            top6.append(idx)
        elif score > 0.4:
            top4.append(idx)
        elif score > 0.2:
            top2.append(idx)
        else:
            top0.append(idx)

    top6 = list(set(top6))
    top4 = list(set(top4))
    top2 = list(set(top2))
    top0 = list(set(top0))

    top6.extend(top4)
    top6.extend(top2)
    top6.extend(top0)

    tmp_list = []
    for idx in top6:
        desc = desc_list[idx]
        tmp_list.append(desc)

    tmp_list = list(set(tmp_list))
    outputs = []
    for i in range(1000):
        desc = tmp_list[i]
        outputs.append([0,0,desc,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_f_{target}.csv', index=False)
    print(f'Saving ../data/eval/eval_f_{target}.csv success!')


def sr_target_qed(fname, fname2, target):
    f =  open(fname,'r')
    lines = f.readlines()

    predicts = []
    truth = []
    desc_list = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

            if 'INFO: 0: ' in line:
                split_token = 'INFO: 0: '
            elif 'INFO: 1: ' in line:
                split_token = 'INFO: 1: '
            elif 'INFO: 2: ' in line:
                split_token = 'INFO: 2: '
            elif 'INFO: 3: ' in line:
                split_token = 'INFO: 3: '
            elif 'INFO: 4: ' in line:
                split_token = 'INFO: 4: '
            else:
                pass

            desc = tmp[0].strip().split(split_token)[-1].strip()
            desc_list.append(desc)

            # break
        elif 'Reference' in line:
            tmp = line.split('Reference smiles: ')[1].strip()
            truth.append(tmp)
        else:
            pass

    nums = len(predicts)
    # print(nums)

    idx_list = []
    for i in range(nums):
        smile = predicts[i]

        try:
            m=Chem.MolFromSmiles(smile)
        except Exception as e:              # 不能解析的话跳过
            continue

        if m is None:
            continue

        qed=QED.qed(m)
        if qed < 0.6:
            continue

        # maccs = maccs_from_mol(m)
        # idx = math.floor(i/25)
        # idx = math.floor(i)
        idx = i
        idx_list.append(idx)

    ##############################################################
    df = pd.read_csv(fname2, sep=' ')
    score_list = df['1'].values
    
    len_of_score_list = len(score_list)
    len_of_idx_list = len(idx_list)

    assert len_of_score_list == len_of_idx_list

    top6 = []
    top4 = []
    top2 = []
    top0 = []
    for i in range(len_of_score_list):
        score = score_list[i]
        idx = idx_list[i]
        if score > 0.6:
            top6.append(idx)
        elif score > 0.4:
            top4.append(idx)
        elif score > 0.2:
            top2.append(idx)
        else:
            top0.append(idx)

    top6 = list(set(top6))
    top4 = list(set(top4))
    top2 = list(set(top2))
    top0 = list(set(top0))

    top6.extend(top4)
    top6.extend(top2)
    top6.extend(top0)

    tmp_list = []
    for idx in top6:
        desc = desc_list[idx]
        tmp_list.append(desc)

    tmp_list = list(set(tmp_list))
    outputs = []
    for i in range(1000):
        desc = tmp_list[i]
        outputs.append([0,0,desc,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_f_{target}.csv', index=False)
    print(f'Saving ../data/eval/eval_f_{target}.csv success!')


def sr_target_qed_sa(fname, fname2, target):
    f =  open(fname,'r')
    lines = f.readlines()

    predicts = []
    truth = []
    desc_list = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

            if 'INFO: 0: ' in line:
                split_token = 'INFO: 0: '
            elif 'INFO: 1: ' in line:
                split_token = 'INFO: 1: '
            elif 'INFO: 2: ' in line:
                split_token = 'INFO: 2: '
            elif 'INFO: 3: ' in line:
                split_token = 'INFO: 3: '
            elif 'INFO: 4: ' in line:
                split_token = 'INFO: 4: '
            else:
                pass

            desc = tmp[0].strip().split(split_token)[-1].strip()
            desc_list.append(desc)

            # break
        elif 'Reference' in line:
            tmp = line.split('Reference smiles: ')[1].strip()
            truth.append(tmp)
        else:
            pass

    nums = len(predicts)
    # print(nums)

    idx_list = []
    for i in range(nums):
        smile = predicts[i]

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

        # maccs = maccs_from_mol(m)
        # idx = math.floor(i/25)
        # idx = math.floor(i)
        idx = i
        idx_list.append(idx)

    ##############################################################
    df = pd.read_csv(fname2, sep=' ')
    score_list = df['1'].values
    
    len_of_score_list = len(score_list)
    len_of_idx_list = len(idx_list)

    assert len_of_score_list == len_of_idx_list

    top6 = []
    top4 = []
    top2 = []
    top0 = []
    for i in range(len_of_score_list):
        score = score_list[i]
        idx = idx_list[i]
        if score > 0.6:
            top6.append(idx)
        elif score > 0.4:
            top4.append(idx)
        elif score > 0.2:
            top2.append(idx)
        else:
            top0.append(idx)

    top6 = list(set(top6))
    top4 = list(set(top4))
    top2 = list(set(top2))
    top0 = list(set(top0))

    top6.extend(top4)
    top6.extend(top2)
    top6.extend(top0)

    tmp_list = []
    for idx in top6:
        desc = desc_list[idx]
        tmp_list.append(desc)

    tmp_list = list(set(tmp_list))
    outputs = []
    for i in range(1000):
        desc = tmp_list[i]
        outputs.append([0,0,desc,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_f_{target}.csv', index=False)
    print(f'Saving ../data/eval/eval_f_{target}.csv success!')

# fname_bbb = '../log/eval_gtp2_292772_bbb_150.log'
# sr_target(fname_bbb, '../log/bbb.txt', 'bbb')

# fname_hia = '../log/eval_gtp2_292772_hia_150.log'
# sr_target(fname_hia, '../log/hia.txt', 'hia')

# fname_pgps = '../log/eval_gtp2_292772_pgps_150.log'
# sr_target(fname_pgps, '../log/pgps.txt', 'pgps')

# fname_bbb = '../log/eval_gtp2_292772_bbb_qed_150.log'
# sr_target_qed(fname_bbb, '../log/bbb_qed.txt', 'bbb_qed')

# fname_hia = '../log/eval_gtp2_292772_hia_qed_150.log'
# sr_target_qed(fname_hia, '../log/hia_qed.txt', 'hia_qed')

# fname_pgps = '../log/eval_gtp2_292772_pgps_qed_150.log'
# sr_target_qed(fname_pgps, '../log/pgps_qed.txt', 'pgps_qed')

fname_bbb = '../log/eval_gtp2_292772_bbb_qed_sa_150.log'
sr_target_qed_sa(fname_bbb, '../log/bbb_qed_sa.txt', 'bbb_qed_sa')

fname_hia = '../log/eval_gtp2_292772_hia_qed_sa_150.log'
sr_target_qed_sa(fname_hia, '../log/hia_qed_sa.txt', 'hia_qed_sa')

fname_pgps = '../log/eval_gtp2_292772_pgps_qed_sa_150.log'
sr_target_qed_sa(fname_pgps, '../log/pgps_qed_sa.txt', 'pgps_qed_sa')

print('Done')