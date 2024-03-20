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


def maccs_from_mol(mol):
    maccs = [1]
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_bits = list(fp.GetOnBits())
    for fp in fp_bits:
        maccs.append(f'{fp}:1')
    return maccs

def sr_target(fname):
    f =  open(fname,'r')
    lines = f.readlines()

    outputs = []
    predicts = []
    truth = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

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

        try:
            m=Chem.MolFromSmiles(smile)
        except Exception as e:              # 不能解析的话跳过
            continue

        if m is None:
            continue

        maccs = maccs_from_mol(m)
        outputs.append(maccs)

    out_fname = fname.split('.log')[0].split('log/')[1]
    print(fname,out_fname)
    outputs = pd.DataFrame(data=outputs)
    outputs.to_csv(f'../log/maccs_{out_fname}.csv', index=False, header=False, sep=' ')
    print(f'Saving ../log/maccs_{out_fname}.csv success!')


def sr_target_qed(fname):
    f =  open(fname,'r')
    lines = f.readlines()

    outputs = []
    predicts = []
    truth = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

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

        try:
            m=Chem.MolFromSmiles(smile)
        except Exception as e:              # 不能解析的话跳过
            continue

        if m is None:
            continue

        qed=QED.qed(m)
        if qed < 0.6:
            continue

        maccs = maccs_from_mol(m)
        outputs.append(maccs)

    out_fname = fname.split('.log')[0].split('log/')[1]
    print(fname,out_fname)
    outputs = pd.DataFrame(data=outputs)
    outputs.to_csv(f'../log/maccs_{out_fname}.csv', index=False, header=False, sep=' ')
    print(f'Saving ../log/maccs_{out_fname}.csv success!')

def sr_target_qed_sa(fname):
    f =  open(fname,'r')
    lines = f.readlines()

    outputs = []
    predicts = []
    truth = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

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

        maccs = maccs_from_mol(m)
        outputs.append(maccs)

    out_fname = fname.split('.log')[0].split('log/')[1]
    print(fname,out_fname)
    outputs = pd.DataFrame(data=outputs)
    outputs.to_csv(f'../log/maccs_{out_fname}.csv', index=False, header=False, sep=' ')
    print(f'Saving ../log/maccs_{out_fname}.csv success!')


# fname_bbb = '../log/eval_gtp2_292772_1_bbb_150.log'
# sr_target(fname_bbb)
# fname_bbb = '../log/eval_gtp2_292772_2_bbb_150.log'
# sr_target(fname_bbb)
# fname_hia = '../log/eval_gtp2_292772_1_hia_150.log'
# sr_target(fname_hia)
# fname_hia = '../log/eval_gtp2_292772_2_hia_150.log'
# sr_target(fname_hia)
# fname_pgps = '../log/eval_gtp2_292772_1_pgps_150.log'
# sr_target(fname_pgps)
# fname_pgps = '../log/eval_gtp2_292772_2_pgps_150.log'
# sr_target(fname_pgps)



# fname_bbb = '../log/eval_gtp2_292772_f_bbb_150.log'
# sr_target(fname_bbb)
# fname_hia = '../log/eval_gtp2_292772_f_hia_150.log'
# sr_target(fname_hia)
# fname_pgps = '../log/eval_gtp2_292772_f_pgps_150.log'
# sr_target(fname_pgps)

# fname_bbb = '../log/eval_gtp2_292772_f_bbb_qed_150.log'
# sr_target_qed(fname_bbb)
# fname_hia = '../log/eval_gtp2_292772_f_hia_qed_150.log'
# sr_target_qed(fname_hia)
# fname_pgps = '../log/eval_gtp2_292772_f_pgps_qed_150.log'
# sr_target_qed(fname_pgps)

# fname_bbb = '../log/eval_gtp2_292772_f_bbb_qed_sa_150.log'
# sr_target_qed_sa(fname_bbb)
fname_hia = '../log/eval_gtp2_292772_f_hia_qed_sa_150.log'
sr_target_qed_sa(fname_hia)
# fname_pgps = '../log/eval_gtp2_292772_f_pgps_qed_sa_150.log'
# sr_target_qed_sa(fname_pgps)

def sr_qed_logp(fname, logp=2):
    f =  open(fname,'r')
    lines = f.readlines()

    outputs = []
    predicts = []
    truth = []
    for line in lines:
        if 'startofsmiles' in line:
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')
            # predicts.append(tmp[1].split(' ')[0].strip())
            predicts.append(tmp[-1].strip())

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

        maccs = maccs_from_mol(m)
        outputs.append(maccs)

    out_fname = fname.split('.log')[0].split('log/')[1]
    print(fname,out_fname)
    outputs = pd.DataFrame(data=outputs)
    outputs.to_csv(f'../log/maccs_{out_fname}.csv', index=False, header=False, sep=' ')
    print(f'Saving ../log/maccs_{out_fname}.csv success!')

# fname_bd = '../log/eval_gtp2_1377731_bd_40.log'
# sr_qed_logp(fname_bd)

# fname_ft = '../log/eval_gtp2_1377731_ft_40.log'
# sr_qed_logp(fname_ft)

# fname_bd_ft = '../log/eval_gtp2_1377731_bd_ft_40.log'
# sr_qed_logp(fname_bd_ft)