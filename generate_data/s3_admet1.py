import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import QED
import sys
import random
from rdkit.Chem import MACCSkeys

# this script is for sentence generate

file_name = '../data/pubchem_100_900.csv'
chunk = pd.read_csv(file_name, iterator=True)

def maccs_from_mol(mol):
    maccs = [0]
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_bits = list(fp.GetOnBits())
    for fp in fp_bits:
        maccs.append(f'{fp}:1')
    return maccs

def generate_desc_by_chunk(chunk_data, outputs):
    
    counts = 0

    for item in chunk_data.values:
        cid = item[0]
        iupac = item[1]
        smiles = item[2]

        try:
            m=Chem.MolFromSmiles(smiles)
        except Exception as e:              # 不能解析的话跳过
            continue

        maccs = maccs_from_mol(m)

        outputs.append(maccs)
        counts = counts + 1

    return counts


switch = True
fid = 0
total_counts = 0
while switch:
    fid = fid + 1

    if fid > 5:
        break

    outputs = []

    counts = 0
    while switch:
        try:
            chunk_data = chunk.get_chunk(10000)
            print("Get chunk: ", counts)
        except Exception as e:
            switch = False
            continue

        counts = generate_desc_by_chunk(chunk_data,outputs) + counts

        if counts >= 500000:
            break

    total_counts = total_counts + counts
    
    outputs = pd.DataFrame(data=outputs)
    outputs.to_csv('../data/maccs_{}.csv'.format(fid), index=False, header=False, sep=' ')
    print("保存 ../data/maccs_{}.csv 成果，共 {} 条数据，累计保存 {} 条数据。".format(fid, counts, total_counts))
print("Done.")