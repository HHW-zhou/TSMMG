import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import QED

def add_qed(target, data, outputs):
    for item in data:
        cid = 'none'
        iupac = 'none'
        smiles = item[1]

        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)               # QED analysis

        if qed > 0.6:
            desc = f'It is active to {target}. Its qed is bigger than 0.6.'
        else:
            desc = f'It is active to {target}. Its qed is smaller than 0.6.'

        outputs.append([cid,iupac,desc,smiles])

targets = ['drd2','jnk3','gsk3']
for target in targets:
    file_name = f'../data/{target}/{target}_active.csv'
    df = pd.read_csv(file_name)
    data = df.values
    outputs = []
    add_qed(target, data, outputs)

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/{target}_qed2.csv', index=False)
print("Done.")