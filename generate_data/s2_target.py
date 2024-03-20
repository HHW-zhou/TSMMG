import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

# 这步之前要先用smiles2iupac反向生成IUPAC

# this script is for sentence generate
t1 = "It is active to {}."
t2 = " It has {} synthetic accessibility."
def generate_desc_of_target(target):
    fname = f'../data/{target}_positive.csv'
    df = pd.read_csv(fname)

    smiles_list = df['smiles'].values[:50000]

    outputs = []
    for smiles in smiles_list:
        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)               # QED analysis

        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score
        sas_degree = 'good' if sas < 4 else 'poor'
        
        desc = t1.format(target.upper())
        
        if qed > 0.6:
            desc2 = desc + ' It has a high qed score.'
        else:
            desc2 = desc + ' It has a low qed score.'

        desc3 = desc2 + t2.format(sas_degree)

        outputs.append([0,None,desc,smiles])
        outputs.append([0,None,desc2,smiles])
        outputs.append([0,None,desc3,smiles])


    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/train/{target}.csv', index=False)
    print(f"Saving ../data/train/{target}.csv done.")

# generate_desc_of_target('drd2')
# generate_desc_of_target('gsk3')
# generate_desc_of_target('jnk3')


tasks = ['drd2','jnk3','gsk3']


dfs = []
for task in tasks:
    fname = f'../data/train/{task}.csv'
    df = pd.read_csv(fname)
    print(f"{task}: {len(df)}")
    dfs.append(df)

merged_df = pd.concat(dfs)
merged_df.to_csv('../data/train/target.csv', index=False)
print("Total number: ", len(merged_df))
print("Done.")