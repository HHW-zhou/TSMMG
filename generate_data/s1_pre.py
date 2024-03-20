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

t1 = "The molecule contains {}."
# t2 = " its molecular weight (MW) is {};"
# t3 = " its logP is {} which {} than 0 which means the molecule is {};"
t3 = " Its logP is {}."
# t4 = " the number of hydrogen bond acceptors (HBA) of it is {};"
# t5 = " the number of hydrogen bond donors (HBD) of it is {};"
# t6 = " the topological molecular polar surface area (PSA) of it is {};"
# t7 = " the number of rotatable bonds (ROTB) of it is {};"
# t8 = " the number of aromatic rings (AROM) of it is {};"
# t9 = " the number of structural alerts is {}."
# t10 = " the Synthetic Accessibility score (SAscore) of it is {}."
t10 = " It has {} synthetic accessibility."

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
        # MW = qed.MW
        ALOGP = qed.ALOGP           # lipophilic if ALOGP > 0 else hydrophobic
        # HBA = qed.HBA
        # HBD = qed.HBD
        # PSA = qed.PSA
        # ROTB = qed.ROTB
        # AROM = qed.AROM
        # ALERTS = qed.ALERTS

        sas = sascorer.calculateScore(m)       # Synthetic Accessibility score
        sas_degree = 'good' if sas < 4 else 'poor'

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        for fg_num in range(1,4):
            try:
                sampled_words = random.sample(filtered_word_list, fg_num)
            except Exception as e:
                continue

            components = ''
            for word in sampled_words:
                components = components + word + ', '
            components = components[:-2]
        
            desc1 = t1.format(components)    #功能团
            outputs.append([cid,iupac,desc1,smiles])

            if fg_num == 1:
                # logp
                desc2 = t1.format(components)    #功能团
                desc2 = desc2 + t3.format(round(ALOGP,2))         #logp
                outputs.append([cid,iupac,desc2,smiles])

                # sas
                # desc3 = t1.format(components)    #功能团
                # desc3 = desc3 + t10.format(round(sas,2))          #sas
                # desc3 = desc3 + t10.format(sas_degree)          #sas
                # outputs.append([cid,iupac,desc3,smiles])

                # logp & sas
                desc4 = t1.format(components)    #功能团
                desc4 = desc4 + t3.format(round(ALOGP,2))         #logp
                desc4 = desc4 + t10.format(sas_degree)          #sas
                outputs.append([cid,iupac,desc4,smiles])


        # # 根据不同的drop生成数据
        # drop_list = [0, 0.2, 0.4, 0.6, 0.8]
        # for drop_ratio in drop_list:
        #     tmp_word_list = filtered_word_list.copy()
        #     ###### drop
        #     word_num = len(tmp_word_list)
        #     drop_num = int(word_num * drop_ratio)
        #     drop_idx = random.sample([i for i in range(word_num)], drop_num)
        #     drop_word_list = [tmp_word_list[idx] for idx in drop_idx]
        #     for dword in drop_word_list:
        #         tmp_word_list.remove(dword)
        #     ######

        #     components = ''
        #     for word in tmp_word_list:
        #         components = components + word + ', '
        #     components = components[:-2]
            
            # desc = t1.format(components)        #功能团
            # desc = desc + t3.format(round(ALOGP,2))         #logp
            # desc = desc + t4.format(HBA)
            # desc = desc + t5.format(HBD)
            # desc = desc + t6.format(round(PSA,2))
            # desc = desc + t7.format(ROTB)
            # desc = desc + t8.format(AROM)
            # desc = desc + t9.format(ALERTS)
            # desc = desc + t10.format(round(sas,2))      #sas

            # outputs.append([cid,iupac,desc,smiles])
        counts = counts + 1
    return counts


switch = True
fid = 0
total_counts = 0
while switch:
    fid = fid + 1

    if fid > 22:
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
    outputs.to_csv('../data/train/{}.csv'.format(fid), index=False)
    print("保存 ../data/train/{}.csv 成果，共 {} 条数据，累计保存 {} 条数据。".format(fid, counts, total_counts))
print("Done.")