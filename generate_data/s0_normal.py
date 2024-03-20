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
import random

# this script is for sentence generate

file_name = '../data/pubchem_100_900.csv'
chunk = pd.read_csv(file_name, iterator=True)

t1 = "The molecule contains {};"
t2 = " its molecular weight (MW) is {};"
# t3 = " its logP is {} which {} than 0 which means the molecule is {};"
t3 = " its logP is {};"
t4 = " the number of hydrogen bond acceptors (HBA) of it is {};"
t5 = " the number of hydrogen bond donors (HBD) of it is {};"
t6 = " the topological molecular polar surface area (PSA) of it is {};"
t7 = " the number of rotatable bonds (ROTB) of it is {};"
t8 = " the number of aromatic rings (AROM) of it is {};"
t9 = " the number of structural alerts is {}."
t10 = " the Synthetic Accessibility score (SAscore) of it is {}."

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

        qed=QED.properties(m)               # QED analysis
        MW = qed.MW
        ALOGP = qed.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic
        HBA = qed.HBA
        HBD = qed.HBD
        PSA = qed.PSA
        ROTB = qed.ROTB
        AROM = qed.AROM
        ALERTS = qed.ALERTS

        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score
        

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>1 and item[0].isnumeric() is False]

        components = ''
        for word in filtered_word_list:
            components = components + word + ', '
        components = components[:-2]
        
        desc = t1.format(components)        #功能团
        desc = desc + t3.format(round(ALOGP,2))     #logp
        desc = desc + t10.format(round(sas,2))      #sas

        outputs.append([cid,iupac,desc,smiles])

        counts = counts + 1
    return counts


switch = True
fid = 0
total_counts = 0
while switch:
    fid = fid + 1

    if fid > 20:
        break

    outputs = []

    counts = 0
    while switch:
        try:
            chunk_data = chunk.get_chunk(100)
        except Exception as e:
            switch = False
            continue

        counts = generate_desc_by_chunk(chunk_data,outputs) + counts

        if counts >= 500000:
            break

    total_counts = total_counts + counts
    
    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv('../data/n{}.csv'.format(fid), index=False)
    print("保存 ../data/{}.csv 成果，共 {} 条数据，累计保存 {} 条数据。".format(fid, counts, total_counts))
print("Done.")