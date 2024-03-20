import pandas as pd
from rdkit import Chem
import re
import random

def generate_vocs():
    df_list = []

    for i in range(1,11):
        fname = f'./data/n{i}.csv'
        df_list.append(pd.read_csv(fname))

    df = pd.concat(df_list)

    data_list = df.values

    vocs = []
    vocs_dict = {}
    for item in data_list:
        iupac = item[1]

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        vocs.extend(filtered_word_list)

        for word in filtered_word_list:
            if word in vocs_dict:
                vocs_dict[word] = vocs_dict[word] + 1
            else:
                vocs_dict[word] = 1

    vocs = list(set(vocs))
    outputs = []
    for word, number in vocs_dict.items():
        outputs.append([word,number])

    outputs.sort(key=lambda x:x[1], reverse=True)

    outputs = pd.DataFrame(data=outputs, columns=['voc','number'])
    outputs.to_csv('./data/voc_all.csv', index=False)


def eval_target(target):
    fname = '../data/voc_all.csv'
    df = pd.read_csv(fname)

    voc_list = []
    for data in df.values:
        freq = data[1]
        voc = data[0]

        # if freq > 90:
        voc_list.append(voc)

    outputs = []
    for fg in voc_list:

        if target in ['drd2','gsk3','jnk3']:
            desc = f"The molecule contains {fg}. It is active to {target.upper()}."
        elif target == 'bbb':
            desc = f"The molecule contains {fg}. It can pass through the blood-brain barrier."
        elif target == 'hia':
            desc = f"The molecule contains {fg}. It can be absorbed by human intestinal."
        elif target == 'pgps':
            desc = f"The molecule contains {fg}. It is a P-glycoprotein substrate."

        outputs.append([0,0,desc,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_{target}.csv', index=False)
    print('Saved.')


eval_target('drd2')
eval_target('gsk3')
eval_target('jnk3')
eval_target('bbb')
eval_target('hia')
eval_target('pgps')


def eval_target2(target, tn=1):
    fname = f'../data/eval/eval_{target}.csv'
    df = pd.read_csv(fname)

    outputs = []
    for item in df.values:
        desc = item[2]

        fg = desc.split('contains ')[1].split('.')[0]

        if 'drd2' in target:
            if tn == 1:
                desc = f"I want a molecule that contains {fg} and active to DRD2."
            elif tn == 2:
                desc = f"Give me a molecule which contains {fg} and active to DRD2."
        elif 'gsk3' in target:
            if tn == 1:
                desc = f"I want a molecule that contains {fg} and active to GSK3."
            elif tn == 2:
                desc = f"Give me a molecule which contains {fg} and active to GSK3."
        elif 'jnk3' in target:
            if tn == 1:
                desc = f"I want a molecule that contains {fg} and active to JNK3."
            elif tn == 2:
                desc = f"Give me a molecule which contains {fg} and active to JNK3."
        elif 'bbb' in target:
            if tn == 1:
                desc = f"I want a molecule that contains {fg} and can pass through the blood-brain barrier."
            elif tn == 2:
                desc = f"Give me a molecule which contains {fg} and can pass through the blood-brain barrier."
        elif 'hia' in target:
            if tn == 1:
                desc = f"I want a molecule that contains {fg} and can be absorbed by human intestinal."
            elif tn == 2:
                desc = f"Give me a molecule which contains {fg} and can be absorbed by human intestinal."
        elif 'pgps' in target:
            if tn == 1:
                desc = f"I want a molecule that contains {fg} and is a P-glycoprotein substrate."
            elif tn == 2:
                desc = f"Give me a molecule which contains {fg} and is a P-glycoprotein substrate."

        outputs.append([0,0,desc,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_{tn}_{target}.csv', index=False)
    print('Saved.')


# eval_target2('f_drd2',1)
# eval_target2('f_gsk3',1)
# eval_target2('f_jnk3',1)
# eval_target2('f_bbb',1)
# eval_target2('f_hia',1)
# eval_target2('f_pgps',1)

# eval_target2('f_drd2',2)
# eval_target2('f_gsk3',2)
# eval_target2('f_jnk3',2)
# eval_target2('f_bbb',2)
# eval_target2('f_hia',2)
# eval_target2('f_pgps',2)


def eval_target_qed(target):
    fname = f'../data/eval/eval_{target}.csv'
    df = pd.read_csv(fname)

    outputs = []
    for item in df.values:
        desc = item[2]

        desc_tmp = desc + ' It has a high qed score.'

        outputs.append([0,0,desc_tmp,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_{target}_qed.csv', index=False)


eval_target_qed('drd2')
eval_target_qed('gsk3')
eval_target_qed('jnk3')
eval_target_qed('bbb')
eval_target_qed('hia')
eval_target_qed('pgps')


def eval_target_qed_sa(target):
    fname = f'../data/eval/eval_{target}_qed.csv'
    df = pd.read_csv(fname)

    outputs = []
    for item in df.values:
        desc = item[2]

        desc_tmp = desc + ' It has good synthetic accessibility.' 

        outputs.append([0,0,desc_tmp,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_{target}_qed_sa.csv', index=False)


eval_target_qed_sa('drd2')
eval_target_qed_sa('gsk3')
eval_target_qed_sa('jnk3')

eval_target_qed_sa('bbb')
eval_target_qed_sa('hia')
eval_target_qed_sa('pgps')


def eval_vocs(fg_num):
    t1 = "The molecule contains {}."

    fname = '../data/n1.csv'
    df = pd.read_csv(fname)

    outputs = []
    desc_list = []
    for item in df.values:
        cid = item[0]
        iupac = item[1]
        smiles = item[3]

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue

        try:
            sampled_words = random.sample(filtered_word_list, fg_num)
        except Exception as e:
            continue

        try:
            m=Chem.MolFromSmiles(smiles)
        except Exception as e:              # 不能解析的话跳过
            continue

        components = ''
        for word in sampled_words:
            components = components + word + ', '
        components = components[:-2]

        desc = t1.format(components)        #功能团

        if desc not in desc_list:
            outputs.append([cid,iupac,desc,smiles])
            desc_list.append(desc)

        if len(outputs) == 10000:
            break

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval_{fg_num}_fg.csv', index=False)
    print('Saved.')

# eval_vocs(1)
# eval_vocs(2)
# eval_vocs(3)


def eval_vocs2(fg_num):
    t1 = "The molecule contains {};"
    t3 = " its logP is {};"
    t10 = " the Synthetic Accessibility score (SAscore) of it is {}."

    fname = '../data/voc_all.csv'
    df = pd.read_csv(fname)

    count = 0
    vocs_list = []
    for data in df.values:
        if data[1] > 500:
            count = count + 1
            vocs_list.append(data[0])

    outputs = []
    desc_list = []
    while True:
        sampled_words = random.sample(vocs_list, fg_num)
        components = ''
        for word in sampled_words:
            components = components + word + ', '
        components = components[:-2]

        desc = t1.format(components)        #功能团
        desc = desc[:-1] + '.'

        if desc not in desc_list:
            outputs.append([0,0,desc,0])
            desc_list.append(desc)

        if len(outputs) == 1000:
            break

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval_{fg_num}_fg2.csv', index=False)
    print('Saved.')


def eval_bd_t(target):
    fname = '../data/n1.csv'
    df = pd.read_csv(fname)

    voc = []
    for item in df.values:
        iupac = item[1]

        word_list = re.split("[\s\[\],\(\)-.;]",iupac)
        filtered_word_list = [item for item in word_list if len(item)>2 and item[0].isnumeric() is False]
        if len(filtered_word_list) == 0:
            continue
        voc.extend(filtered_word_list)

    descs = []
    used_fg = []
    while True:
        fg = random.choice(voc)

        # if fg in trained_vocs:
        #     continue

        if fg in used_fg:
            continue

        if target == 'bd':
            desc = f"The molecule contains {fg}. It has a high qed score. Its logP is 2. It is ready biodegradable."
        elif target == 'ft':
            desc = f"The molecule contains {fg}. It has a high qed score. Its logP is 2. It has low fish toxicity."
        elif target == 'bd_ft':
            desc = f"The molecule contains {fg}. It has a high qed score. Its logP is 2. It is ready biodegradable. It has low fish toxicity."
        else:
            pass

        descs.append(desc)
        used_fg.append(fg)

        if len(descs) == 1000:
            break

        # print('length: ', len(descs))

    outputs = []
    for desc in descs:
        outputs.append([0,0,desc,0])

    outputs = pd.DataFrame(data=outputs, columns=['cid', 'iupac', 'desc', 'smiles'])
    outputs.to_csv(f'../data/eval/eval_{target}.csv', index=False)
    print('Saved.')

# eval_bd_t('bd')
# eval_bd_t('ft')
# eval_bd_t('bd_ft')