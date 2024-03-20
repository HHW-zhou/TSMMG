import sys
import rdkit
from argparse import ArgumentParser
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import pandas as pd

################################ 训练集数据 #######################################
true_mols = []

# true_fname = '../data/train/bbb_qed.csv'
# true_fname = '../data/train/hia_qed.csv'
true_fname = '../data/train/pgps_qed.csv'

true_df = pd.read_csv(true_fname)
true_list = true_df.values[:80000]
for item in true_list:
    smile = item[3]
    try:
        m = Chem.MolFromSmiles(smile)
    except Exception as e:
        continue

    if m is None:
        continue

    true_mols.append(m)
true_mols = list(set(true_mols))
###############################################################################

################################### 预测数据 ###################################
pred_mols = []

# fname = '../log/eval_gtp2_292772_f_bbb_qed_sa_150.log'
# fname = '../log/eval_gtp2_292772_f_hia_qed_sa_150.log'
fname = '../log/eval_gtp2_292772_f_pgps_qed_sa_150.log'
f =  open(fname,'r')
lines = f.readlines()
predicts = []
for line in lines:
    if 'startofsmiles' in line:
        # tmp = line.split('>')[1].strip()
        tmp = line.split('<|startofsmiles|>')
        # predicts.append(tmp[1].split(' ')[0].strip())
        predicts.append(tmp[-1].strip())

nums = len(predicts)
for i in range(nums):
    smile = predicts[i]
    try:
        m = Chem.MolFromSmiles(smile)
    except Exception as e:
        continue

    if m is None:
        continue

    pred_mols.append(m)
###############################################################################
true_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in true_mols]
pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

fraction_similar = 0

sim_distribution = []
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)

    if max(sims) >= 0.4:
        fraction_similar += 1
    sim_distribution.append(max(sims))

print('novelty:', 1 - fraction_similar / len(pred_mols))

similarity = 0
for i in range(len(pred_fps)):
    sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
    similarity += sum(sims)

n = len(pred_fps)
n_pairs = n * (n - 1) / 2
diversity = 1 - similarity / n_pairs
print('diversity:', diversity)


moses_count = 0
for smile in predicts:
    if smile not in true_list:
        moses_count = moses_count + 1

print('moses novelty:', moses_count/nums)