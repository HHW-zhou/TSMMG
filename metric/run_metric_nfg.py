import sys
sys.path.append("../")

from metric import metric, calculate_diversity
from utils import get_parse
import pandas as pd

args = get_parse()

backbone = args.backbone
training_strategy = args.training_strategy
checkpoint = args.checkpoint
eval_type = args.eval_type
file_name = f'../outputs/output_{backbone}_{training_strategy}_{eval_type}.csv'

task_list = ['IsValid']
target_list = ['Valid']

target_tasks = ['drd2','gsk3']
admet_tasks = ['bbb','hia']

task = 'normal'
for t in target_tasks:
    if t in file_name:
        task = 'target'
        task_list.append(t.upper())
        target_list.append('CAN')

for t in admet_tasks:
    if t in file_name:
        task = 'admet'
        task_list.append(t.upper())
        target_list.append('CAN')

if '_qed' in file_name:
    task_list.append('QED')
    target_list.append('HIGH')

if '_sa' in file_name:
    task_list.append('SAs')
    target_list.append('HIGH')

if 'logp' in file_name:
    task_list.append('LogP')
    # condition = int(file_name.split('.csv')[0].split('logp_')[-1])
    condition = file_name.split('.csv')[0].split('logp_')[-1]
    target_list.append(condition)

valid_ratio, unique_ratio, novel_ratio, success_ratio, diversity = metric(file_name, task, task_list, target_list)
print('\n')
print("=======================================================")
print("File: ", file_name)
print("task: ", task)
print("task_list: ", ','.join(task_list))
print("target_list: ", ','.join(target_list))
print(f"Valid: {round(valid_ratio*100,2)}, Unique: {round(unique_ratio*100,2)}, Novelty: {round(novel_ratio*100,2)}, Diversity: {round(diversity*100,2)}, SR: {round(success_ratio*100,2)}")
print()

model_info = f"{backbone}_{training_strategy}_{checkpoint}_{eval_type}"
task_info = ','.join(task_list)
target_info = ','.join(target_list)
result = []
# result.append([model_info,task_info,target_info,round(valid_ratio*100,2),round(unique_ratio*100,2),round(novel_ratio*100,2),round(diversity*100,2),round(success_ratio*100,2)])
result.append([model_info,task_info,target_info,valid_ratio,unique_ratio,novel_ratio,diversity,success_ratio])
outputs = pd.DataFrame(data=result, columns=['model_info', 'task_info', 'target_info', 'valid','unique','novel','diversity','SR'])
outputs.to_csv(f'./result/{model_info}-{task_info}.csv', index=False)