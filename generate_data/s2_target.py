import os
import re
import sys
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

# 这步之前要先用smiles2iupac反向生成IUPAC

# this script is for sentence generate
t0 = "The molecule contains {}. It is active to {}."
t1 = "The molecule contains {}. It is active to {}. It has a {} qed score."
t2 = "The molecule contains {}. It is active to {}. It has a {} qed score. It has {} synthetic accessibility."

def generate_desc_of_target(target):
    fname = f'../data/{target}_positive.csv'
    df = pd.read_csv(fname)

    outputs = []
    for i in range(len(df)):
        smiles = df['smiles'][i]
        iupac = df['iupac'][i]

        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)               # QED analysis
        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score

        qed_degree = 'high' if qed > 0.6 else 'low'
        sas_degree = 'good' if sas < 4 else 'poor'

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        used_words = []
        for i in range(4):
            sampled_words = random.sample(filtered_word_list, 1)
            if sampled_words in used_words:
                continue

            components = ''
            for word in sampled_words:
                components = components + word + ', '
            components = components[:-2]

            desc0 = t0.format(components, target.upper())
            desc1 = t1.format(components, target.upper(),qed_degree)
            desc2 = t2.format(components, target.upper(),qed_degree,sas_degree)

            outputs.append([0,iupac,desc0,smiles])
            outputs.append([0,iupac,desc1,smiles])
            outputs.append([0,iupac,desc2,smiles])

            used_words.append(sampled_words)


    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/train/{target}.csv', index=False)
    print(f"Saving ../data/train/{target}.csv done.")

generate_desc_of_target('drd2')
generate_desc_of_target('gsk3')
generate_desc_of_target('jnk3')


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