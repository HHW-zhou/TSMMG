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

############# common properties
def get_mols(smiles_list):
    mol_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)

    return mol_list

def get_valid(mol_list):
    valid_list = []

    for mol in mol_list:
        if mol is None:
            valid_list.append(False)
        else:
            valid_list.append(True)

    return valid_list

def get_qeds(mol_list):
    qed_list = []

    for mol in mol_list:
        if mol is None:
            qed = -19
        else:
            qed=QED.qed(mol)
            qed = round(qed,2)

        qed_list.append(qed)
    return qed_list

def get_sass(mol_list):
    sas_list = []

    for mol in mol_list:
        if mol is None:
            SAs = -19
        else:
            SAs = sascorer.calculateScore(mol)
            SAs = round(SAs,2)

        sas_list.append(SAs)
    return sas_list

def get_logps(mol_list):
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

def get_bbbs(mol_list):
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

def get_hias(mol_list):
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

def get_drd2s(mol_list):
    clf_path = f'../target/drd2.pkl'

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    scores = get_scores(clf, mol_list)
    return scores

def get_gsk3s(mol_list):
    clf_path = f'../target/gsk3.pkl'

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    scores = get_scores(clf, mol_list)
    return scores
#--------------------------------------------------#

###### driver
def get_properties(mol_list, task_list):
    property_dict = {}

    func_map = {
        'IsValid':get_valid,
        'QED':get_qeds,
        'LogP':get_logps,
        'SAs':get_sass,
        'BBB':get_bbbs,
        'HIA':get_hias,
        'DRD2':get_drd2s,
        'GSK3':get_gsk3s
    }

    for task in task_list:
        func = func_map[task]
        labels = func(mol_list)
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

def check_logp(df, condition=1):
    LogP = df['LogP']

    diff = abs(LogP - condition)

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

def sr_check(df, task_list:list, target_list:list):
    func_map = {
        'IsValid':check_valid,
        'QED':check_qed,
        'LogP':check_logp,
        'SAs':check_sas,
        'BBB':check_bbb,
        'HIA':check_hia,
        'DRD2':check_drd2,
        'GSK3':check_gsk3
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

    property_dict = get_properties(mol_list, task_list)
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

    return valid_ratio, unique_ratio, novel_ratio, success_ratio

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
            # tmp = line.split('>')[1].strip()
            tmp = line.split('<|startofsmiles|>')[-1].strip()
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
    
    nums = len(predicts)
    # print(nums)

    predict_iupac_list = []
    for i in tqdm(range(nums)):
        # print(predicts[i])
        SMILES = predicts[i]
        m = Chem.MolFromSmiles(SMILES)
        if m is not None:                                   #如果预测的SMILES有效，则计算IUPAC_NAME
            IUPAC_name = translate_forward(SMILES)
        else:
            IUPAC_name = 'Invalid'
        predict_iupac_list.append(IUPAC_name)

    return truth, voc_list, predicts, predict_iupac_list

def get_train_smiles(task):
    if task == 'normal':
        df_list = []
        for i in range(1,11):
            fname = f"../data/train/{i}.csv"
            df = pd.read_csv(fname)
            df_list.append(df)

        df = pd.concat(df_list)
    else: 
        fname = f"../data/train/{task}.csv"
        df = pd.read_csv(fname)

    train_smiles = df['smiles'].values
    return train_smiles