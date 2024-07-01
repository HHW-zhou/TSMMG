from rdkit import Chem, DataStructs
from rdkit.Chem import QED, MACCSkeys, AllChem
import pandas as pd
# from STOUT import translate_forward, translate_reverse
from tqdm import tqdm
import time
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!

import sascorer

from libsvm.svmutil import *
import pickle
import numpy as np
import torch

############# common properties
def get_mols(smiles_list):
    mol_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)

    return mol_list

def get_valid(df, mol_list):
    valid_list = []

    for mol in mol_list:
        if mol is None:
            valid_list.append(False)
        else:
            valid_list.append(True)

    return valid_list

def get_fg_status(df, mol_list):
    prompt = df['prompt']
    fgs = df['fgs']

    # print("====> ", prompt)
    # print("====> ", fgs)

    fg_status_list = []
    for i in range(len(prompt)):
        fg_list = prompt[i].split('contains')[-1].split('.')[0].split(',')
        status = True
        for fg in fg_list:
            fg = fg.strip()
            # print(fg)

            if fg in ''.join(fgs[i]):
                status = status & True
            else:
                status = status & False

        # print(status, fg_list, ''.join(fgs[i]))
        fg_status_list.append(status)

    return fg_status_list

def get_qeds(df, mol_list):
    qed_list = []

    for mol in mol_list:
        if mol is None:
            qed = -19
        else:
            qed=QED.qed(mol)
            qed = round(qed,2)

        qed_list.append(qed)
    return qed_list

def get_sass(df, mol_list):
    sas_list = []

    for mol in mol_list:
        if mol is None:
            SAs = -19
        else:
            SAs = sascorer.calculateScore(mol)
            SAs = round(SAs,2)

        sas_list.append(SAs)
    return sas_list

def get_logps(df, mol_list):
    logp_list = []

    for mol in mol_list:
        if mol is None:
            LogP = -19
        else:
            qed_=QED.properties(mol)
            LogP = qed_.ALOGP
            LogP = round(LogP,2)

        logp_list.append(LogP)
    return logp_list
#--------------------------------------------------#

############# ADMET
def maccs_from_mol(mol):
    maccs = {}

    if mol is None:
        return maccs

    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_bits = list(fp.GetOnBits())
    for fp in fp_bits:
        maccs[fp] = 1
    return maccs

def get_bbbs(df, mol_list):
    maccs_list = []
    y = []
    for mol in mol_list:
        maccs = maccs_from_mol(mol)
        maccs_list.append(maccs)
        y.append(1)

    model = svm_load_model(f'../ADMET/models/A_BBB_I')
    labels = model.get_labels()
    idx = labels.index(1)
    p_label, p_acc, p_val = svm_predict(y, maccs_list, model, '-b 1')

    # return p_label
    return [i[idx] for i in p_val]

def get_hias(df, mol_list):
    maccs_list = []
    y = []
    for mol in mol_list:
        maccs = maccs_from_mol(mol)
        maccs_list.append(maccs)
        y.append(1)

    model = svm_load_model(f'../ADMET/models/A_HIA_I')
    labels = model.get_labels()
    idx = labels.index(1)
    p_label, p_acc, p_val = svm_predict(y, maccs_list, model, '-b 1')

    # return p_label
    return [i[idx] for i in p_val]
#--------------------------------------------------#

########## targets
def fingerprints_from_mol(mol):  # use ECFP4
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features.reshape(1, -1)

def get_scores(clf, mol_list):
    fps = []
    mask = []
    for mol in mol_list:
        mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
        fps.append(fp)

    fps = np.concatenate(fps, axis=0)
    scores = clf.predict_proba(fps)[:, 1]
    scores = scores * np.array(mask)
    return np.float32(scores)

def get_drd2s(df, mol_list):
    clf_path = f'../target/drd2.pkl'

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    scores = get_scores(clf, mol_list)
    return scores

def get_gsk3s(df, mol_list):
    clf_path = f'../target/gsk3.pkl'

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    scores = get_scores(clf, mol_list)
    return scores
#--------------------------------------------------#

###### driver
def get_properties(df, mol_list, task_list):
    property_dict = {}

    func_map = {
        'IsValid':get_valid,
        'QED':get_qeds,
        'LogP':get_logps,
        'SAs':get_sass,
        'BBB':get_bbbs,
        'HIA':get_hias,
        'DRD2':get_drd2s,
        'GSK3':get_gsk3s,
        'FG':get_fg_status
    }

    for task in task_list:
        func = func_map[task]
        labels = func(df, mol_list)
        property_dict[task] = labels

    return property_dict
#--------------------------------------------------#

######################### checker #############################
def check_valid(df, condition='HIGH'):
    valid = df['IsValid']

    return valid

def check_qed(df, condition='HIGH'):
    qed = df['QED']

    if condition == 'HIGH':
        return qed > 0.6
    elif condition == 'LOW':
        return qed <= 0.6
    elif condition == 'ANY':
        return qed > -99

def check_logp(df, condition='1'):
    LogP = df['LogP']

    diff = abs(LogP - int(condition))

    return diff < 1

def check_sas(df, condition='HIGH'):
    SAs = df['SAs']

    if condition == 'HIGH':             # <4 高可合成性
        return SAs < 4
    elif condition == 'LOW':
        return SAs >= 4
    elif condition == 'ANY':
        return SAs > -99

def check_bbb(df, condition = 'CAN'):
    bbb = df['BBB']

    if condition == 'CAN':
        # return bbb == 1
        return bbb > 0.5
    elif condition == 'CANNOT':
        # return bbb == 0
        return bbb <= 0.5
    
def check_hia(df, condition = 'CAN'):
    hia = df['HIA']

    if condition == 'CAN':
        # return hia == 1
        return hia > 0.5
    elif condition == 'CANNOT':
        # return hia == 0
        return hia <= 0.5

def check_drd2(df, condition = 'CAN'):
    drd2 = df['DRD2']

    if condition == 'CAN':
        return drd2 > 0.5
    elif condition == 'CANNOT':
        return drd2 <= 0.5
    
def check_gsk3(df, condition = 'CAN'):
    gsk3 = df['GSK3']

    if condition == 'CAN':
        return gsk3 > 0.5
    elif condition == 'CANNOT':
        return gsk3 <= 0.5
    
def check_fg(df, condition = 'IN'):
    FG_status = df['FG']

    return FG_status

def sr_check(df, task_list:list, target_list:list):
    func_map = {
        'IsValid':check_valid,
        'QED':check_qed,
        'LogP':check_logp,
        'SAs':check_sas,
        'BBB':check_bbb,
        'HIA':check_hia,
        'DRD2':check_drd2,
        'GSK3':check_gsk3,
        'FG':check_fg
    }

    result = True
    for i in range(len(task_list)):
        func_name = task_list[i]
        condition = target_list[i]
        ret = func_map[func_name](df, condition)
        # ret = func_map[func_name](item)
        result = result & ret
    return result
#--------------------------------------------------------#

def get_novels(smiles_list, train_smiles):
    novel_list = []

    for smiles in smiles_list:
        if smiles not in train_smiles:
            novel_list.append(True)
        else:
            novel_list.append(False)

    return novel_list

def metric(fname, task, task_list, target_list):
    train_smiles = get_train_smiles(task)

    df = pd.read_csv(fname)
    len_of_data = len(df)

    smiles_list = df['smiles'].values
    mol_list = get_mols(smiles_list)

    property_dict = get_properties(df, mol_list, task_list)
    for key, value in property_dict.items():
        df[key] = value

    ##################### novelty ######################
    novel_list = get_novels(smiles_list, train_smiles)
    df['IsNovel'] = novel_list
    #--------------------------------------------------#

    ##################### check success ######################
    success_list = sr_check(df, task_list, target_list)
    df['IsSuccess'] = success_list
    #--------------------------------------------------------#

    mask = df['IsValid']
    valid_list = df['smiles'][mask]

    valid_num = len(valid_list)
    valid_ratio = round(valid_num/len_of_data,4)

    unique_num = len(set(valid_list))
    unique_ratio = round(unique_num/valid_num,4)

    novel_num = sum(novel_list)
    novel_ratio = round(novel_num/len_of_data,4)

    success_num = sum(success_list)
    success_ratio = round(success_num/len_of_data,4)

    #################### save ##################
    f_split = fname.split('.')
    sv_fname = '.' + ''.join(f_split[:-1]) + '_mt.csv'
    df.to_csv(sv_fname, index=False)
    #----------------------------------------- #

    diversity = calculate_diversity(sv_fname)

    return valid_ratio, unique_ratio, novel_ratio, success_ratio, diversity

def get_train_smiles(task):
    task_list = ['1','2','3','4','5','target','admet','BTK','3CL','FGFR4','KPCD3','EP2','EP4']
    df_list = []
    for task in task_list:
        fname = f"../data/train/{task}.csv"
        tmp_df = pd.read_csv(fname)
        df_list.append(tmp_df)

    df = pd.concat(df_list)

    train_smiles = list(set(df['smiles'].values))
    return train_smiles


def intra_similarity(smiles_list):
    fp_targets = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, 2048) for smiles in smiles_list]
    # fp_targets = [AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
    # fp_targets = [AllChem.RDKFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]

    # 先计算cyclic组内相似度最小的
    sim_list = []
    for i in range(len(fp_targets)):
        # sims = DataStructs.DiceSimilarity(fp_targets[i], fp_targets)
        sims = DataStructs.BulkTanimotoSimilarity(fp_targets[i], fp_targets)
        # print(sims)
        sim_mean = (np.sum(sims) - 1)/(len(fp_targets)-1)
        sim_list.append(sim_mean)

    sim_tensor = torch.tensor(sim_list)

    sim_max = sim_tensor.max().item()
    sim_min = sim_tensor.min().item()
    sim_avg = sim_tensor.mean().item()

    return sim_max, sim_min, sim_avg


def calculate_diversity(fname):
    df = pd.read_csv(fname)[:5000]
    mask = df['IsValid']
    valid_smiles_list = df['smiles'][mask]
    sim_max, sim_min, sim_mean = intra_similarity(valid_smiles_list)
    return 1 - sim_mean

def reward(df, task, task_list, target_list, rewarded_smiles):
    scores = []

    train_smiles = get_train_smiles(task)
    
    smiles_list = df['smiles'].values
    mol_list = get_mols(smiles_list)

    property_dict = get_properties(df, mol_list, task_list)
    for key, value in property_dict.items():
        df[key] = value

    novel_list = get_novels(smiles_list, train_smiles)
    success_list = sr_check(df, task_list, target_list)

    for i in range(len(novel_list)):
        isNovel = novel_list[i]
        isSuccess = success_list[i]
        smiles = smiles_list[i]

        score = 0.4
        if isNovel:
            score = score + 0.5

        if smiles not in rewarded_smiles:
            score = score + 0.1

        if not isSuccess:
            score = 0

        scores.append(torch.tensor(score))

    return scores