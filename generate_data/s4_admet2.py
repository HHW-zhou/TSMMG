import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import QED
import random

def admet(source_data,fname,code,template):
    df = pd.read_csv(fname, sep=' ')
    data = df.values

    len_of_source_data = len(source_data)
    len_of_data = len(data)

    assert len_of_source_data == len_of_data

    positives = []
    negtives = []

    for i in range(len_of_data):
        #'cid', 'iupac', 'desc', 'smiles'
        # cid = source_data[i][0]
        # iupac = source_data[i][1]
        # desc = source_data[i][2]
        smiles = source_data[i][3]

        score = data[i][1]

        if score > 0.6:
            positives.append(smiles)
        else:
            negtives.append(smiles)

    
    try:
        selected_positives = random.sample(positives,25000)
    except Exception as e:
        selected_positives = positives

    try:
        selected_negtives = random.sample(negtives,25000)
    except Exception as e:
        selected_negtives = negtives


    outputs = []
    for smiles in selected_positives:
        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)               # QED analysis

        desc = template[1]
        if qed > 0.6:
            desc = desc + ' It has a high qed score.'
        else:
            desc = desc + ' It has a low qed score.'

        outputs.append([0,0,desc,smiles])

    for smiles in selected_negtives:
        m=Chem.MolFromSmiles(smiles)
        qed=QED.qed(m)               # QED analysis

        desc = template[0]
        if qed > 0.6:
            desc = desc + ' It has a high qed score.'
        else:
            desc = desc + ' It has a low qed score.'

        outputs.append([0,0,desc,smiles])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/train/{code}.csv', index=False)
    print(f"Saved ../data/train/{code}.csv")
    print(f"Numbers: {len(outputs)}")
    print(f"Positives: {len(selected_positives)}")
    print(f"Negtives: {len(selected_negtives)}")


source_file = '../data/n1.csv'
df_source = pd.read_csv(source_file)
source_data = df_source.values

# 'It can pass through the blood-brain barrier.'
# 'It can be absorbed by human intestinal.'
# 'It is a P-glycoprotein substrate.'
# ' It has a high qed score.'
# ' It has a low qed score.'

# admet(source_data, '../data/admet/A_BBB_I.txt', 'bbb', 
#       ['It can not pass through the blood-brain barrier.','It can pass through the blood-brain barrier.'])

# admet(source_data, '../data/admet/A_Caco2_I.txt', 'caco2', 
#       ['Its Caco-2 permeability is low.','Its Caco-2 permeability is high.'])

# admet(source_data, '../data/admet/A_HIA_I.txt', 'hia', 
#       ['It cannot be absorbed by the human intestinal tract.','It can be absorbed by the human intestinal tract.'])

# admet(source_data, '../data/admet/A_PgpS_I.txt', 'pgps', 
#       ['It is not a P-glycoprotein substrate.','It is a P-glycoprotein substrate.'])

# admet(source_data, '../data/admet/A_PgpI_I.txt', 'pgpiI', 
#       ['It is not a P-glycoprotein inhibitor I.','It is a P-glycoprotein inhibitor I.'])

# admet(source_data, '../data/admet/A_PgpI_II.txt', 'pgpiII', 
#       ['It is not a P-glycoprotein inhibitor II.','It is a P-glycoprotein inhibitor II.'])

# admet(source_data, '../data/admet/M_CYP2C9S_I.txt', 'cyp2c9s', 
#       ['It is not a CYP450 2C9 substrate.','It is a CYP450 2C9 substrate.'])

# admet(source_data, '../data/admet/M_CYP2D6S_I.txt', 'cyp2d6s', 
#       ['It is not a CYP450 2D6 substrate.','It is a CYP450 2D6 substrate.'])

# admet(source_data, '../data/admet/M_CYP3A4S_I.txt', 'cyp3a4s', 
#       ['It is not a CYP450 3A4 substrate.','It is a CYP450 3A4 substrate.'])

# admet(source_data, '../data/admet/M_CYP1A2I_I.txt', 'cyp1a2i', 
#       ['It is not a CYP450 1A2 inhibitor.','It is a CYP450 1A2 inhibitor.'])

# admet(source_data, '../data/admet/M_CYP2C9I_I.txt', 'cyp2c9i', 
#       ['It is not a CYP450 2C9 inhibitor.','It is a CYP450 2C9 inhibitor.'])

# admet(source_data, '../data/admet/M_CYP2C19I_I.txt', 'cyp2c19i', 
#       ['It is not a CYP450 2C19 inhibitor.','It is a CYP450 2C19 inhibitor.'])

# admet(source_data, '../data/admet/M_CYP2D6I_I.txt', 'cyp2d6i', 
#       ['It is not a CYP450 2D6 inhibitor.','It is a CYP450 2D6 inhibitor.'])

# admet(source_data, '../data/admet/M_CYP3A4I_I.txt', 'cyp3a4i', 
#       ['It is not a CYP450 3A4 inhibitor.','It is a CYP450 3A4 inhibitor.'])

# admet(source_data, '../data/admet/M_CYPPro_I.txt', 'cyppro', 
#       ['Its CYP450 inhibitory promiscuity level is low.','Its CYP450 inhibitory promiscuity level is high.'])

# admet(source_data, '../data/admet/M_BIO_I.txt', 'bio', 
#       ['It is not ready biodegradable.','It is ready biodegradable.'])

# admet(source_data, '../data/admet/E_OCT2I_I.txt', 'oct2', 
#       ['It is not a renal organic cation transporter inhibitor.','It is a renal organic cation transporter inhibitor.'])

# admet(source_data, '../data/admet/T_AMES_I.txt', 'ames', 
#       ['It is Non-Ames toxic.','It is Ames toxic.'])

# admet(source_data, '../data/admet/T_Carc_I.txt', 'carc', 
#       ['It is not a carcinogen.','It is a carcinogen.'])

# admet(source_data, '../data/admet/T_hERG_I.txt', 'hergI', 
#       ['It is not a hERG inhibitor I.','It is a hERG inhibitor I.'])

# admet(source_data, '../data/admet/T_hERG_II.txt', 'hergII', 
#       ['It is not a hERG inhibitor II.','It is a hERG inhibitor II.'])

# admet(source_data, '../data/admet/T_FHMT_I.txt', 'fhmt', 
#       ['It has low fish toxicity.','It has high fish toxicity.'])

# admet(source_data, '../data/admet/T_HBT_I.txt', 'hbt', 
#       ['It has low honey bee toxicity.','It has high honey bee toxicity.'])

# admet(source_data, '../data/admet/T_TPT_I.txt', 'tpt', 
#       ['It has low tetrahymena pyriformis toxicity.','It has high tetrahymena pyriformis toxicity.'])



# codes = ['bbb','caco2','hia','pgps','pgpiI','pgpiII','cyp2c9s','cyp2d6s','cyp3a4s','cyp1a2i','cyp2c9i','cyp2c19i',
#          'cyp2d6i','cyp3a4i','cyppro','bio','oct2','ames','carc','hergI','hergII','fhmt','hbt','tpt']

codes = ['bbb','hia','pgps','bio','fhmt']

dfs = []
for code in codes:
    fname = f'../data/train/{code}.csv'
    df = pd.read_csv(fname)
    dfs.append(df)

merged_df = pd.concat(dfs)
merged_df.to_csv('../data/train/admet2.csv', index=False)
print("Total number: ", len(merged_df))
print("Done.")